"""Shared helpers for the WIDS project: settings, data loading, features, LSTM.

Folders are set with environment variables so the code runs anywhere:
  WIDS_DATA_DIR   where the dataset CSVs live (default: data)
  WIDS_OUT_DIR    where results (pictures, logs) are written (default: results)
  WIDS_MODEL_DIR  where the trained model is kept (default: models)
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATA_DIR = os.environ.get("WIDS_DATA_DIR", "data")
OUT_DIR = os.environ.get("WIDS_OUT_DIR", "results")
MODEL_DIR = os.environ.get("WIDS_MODEL_DIR", "models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CLASSES = ["Normal", "Disconnect", "BeaconFlood"]   # Deauth + Disassoc are merged
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
SEQ_LEN = 10        # the LSTM looks at the last 10 seconds
BLOCK = 20          # seconds per block when splitting data fairly
AUG_COPIES = 4      # weakened copies of the attacks added to the training data

GENERAL = ["total_frames", "mean_len", "std_len", "mean_rssi", "std_rssi"]
COMPACT = ["disc", "disc_frac", "sub_8", "beacon_frac", "total_frames"]


def load_windows():
    """Load windows_all.csv (made by make_windows.py) and add the extra columns."""
    w = pd.read_csv(os.path.join(DATA_DIR, "windows_all.csv"))
    w = w.sort_values(["source", "frame.time_epoch"]).reset_index(drop=True)
    # block id = every 20 consecutive seconds of one capture (used for fair splits)
    w["block"] = w["source"] + "_" + (w.groupby("source").cumcount() // BLOCK).astype(str)
    # merge the two look-alike attacks into one class
    w["label3"] = w["label"].replace({"Deauth": "Disconnect", "Disassoc": "Disconnect"})
    for c in ["sub_8", "sub_10", "sub_12"]:      # frame-type counts (8=beacon, 10=disassoc, 12=deauth)
        if c not in w.columns:
            w[c] = 0
    w["disc"] = w["sub_10"] + w["sub_12"]                      # disconnect frames per second
    w["disc_frac"] = w["disc"] / w["total_frames"]             # their share of all frames
    w["beacon_frac"] = w["sub_8"] / w["total_frames"]          # beacons' share of all frames
    return w


def sub_columns(w):
    """Names of all the per-frame-type count columns (sub_8, sub_12, ...)."""
    return [c for c in w.columns if c.startswith("sub_")]


def seqs_from(df, feats):
    """For every second, collect that second plus the 9 before it (same capture)."""
    X = np.zeros((len(df), SEQ_LEN, len(feats)), dtype="float32")
    for _, g in df.groupby("source"):
        vals = np.log1p(g[feats].clip(lower=0).values.astype("float32"))  # tame huge counts
        padded = np.vstack([np.repeat(vals[:1], SEQ_LEN - 1, axis=0), vals])
        for k, idx in enumerate(g.index):
            X[idx] = padded[k:k + SEQ_LEN]
    return X


def augmented_copy(df, rng):
    """Weaken the attacks: shrink disconnect frames by a random factor (3%-100%) per block.
    Only ever used on TRAINING data."""
    a = df.copy()
    factors = {b: np.exp(rng.uniform(np.log(0.03), 0.0)) for b in a["block"].unique()}
    f = a["block"].map(factors)
    is_attack = a["label3"] == "Disconnect"
    new_disc = np.where(is_attack, a["disc"] * f, a["disc"])
    a["total_frames"] = a["total_frames"] - a["disc"] + new_disc
    a["disc"] = new_disc
    a["disc_frac"] = a["disc"] / a["total_frames"]
    a["beacon_frac"] = a["sub_8"] / a["total_frames"]
    return a


class Net(nn.Module):
    """LSTM reads the sequence; a linear layer turns its memory into 3 class scores."""
    def __init__(self, n_feat):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, 32, batch_first=True)
        self.out = nn.Linear(32, len(CLASSES))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.out(h[-1])


def train_net(Xtr, ytr, epochs=40):
    """Train an LSTM. Returns the network plus the scaling numbers (mean, std)."""
    mu = Xtr.reshape(-1, Xtr.shape[2]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[2]).std(0) + 1e-6
    Xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    counts = np.bincount(ytr, minlength=len(CLASSES))
    weights = torch.tensor(len(ytr) / (len(CLASSES) * np.maximum(counts, 1)),
                           dtype=torch.float32)             # rare classes count more
    net = Net(Xtr.shape[2])
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    lossf = nn.CrossEntropyLoss(weight=weights)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 64):                   # mini-batches of 64
            idx = perm[i:i + 64]
            opt.zero_grad()
            lossf(net(Xt[idx]), yt[idx]).backward()          # measure error, adjust
            opt.step()
    net.eval()
    return net, mu, sd


def predict_net(net, mu, sd, X):
    """Class number (0,1,2) for every sequence in X."""
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    with torch.no_grad():
        return net(Xt).argmax(1).numpy()


def fit_predict(Xtr, ytr, Xte, epochs=40):
    net, mu, sd = train_net(Xtr, ytr, epochs)
    return predict_net(net, mu, sd, Xte)
