"""Step 1: turn the raw per-frame CSVs into one row per second ("windows").

Input : Deauth.csv, Disasso.csv, BeaconFlood.csv  (in WIDS_DATA_DIR)
Output: windows_all.csv                           (in WIDS_DATA_DIR)
Run   : python src/make_windows.py
"""
import os
import pandas as pd
from wids_common import DATA_DIR

FILES = ["Deauth.csv", "Disasso.csv", "BeaconFlood.csv"]
COLS = ["frame.time_epoch", "frame.len", "radiotap.dbm_antsignal",
        "wlan.fc.type_subtype", "Label"]


def build_windows(path):
    df = pd.read_csv(path, usecols=COLS, low_memory=False)
    g = df.groupby("frame.time_epoch")          # frames that share the same second

    w = pd.DataFrame({
        "total_frames": g.size(),
        "mean_len": g["frame.len"].mean(),
        "std_len": g["frame.len"].std(),
        "mean_rssi": g["radiotap.dbm_antsignal"].mean(),
        "std_rssi": g["radiotap.dbm_antsignal"].std(),
    })
    # how many frames of each type per second (columns sub_8, sub_12, ...)
    counts = pd.crosstab(df["frame.time_epoch"],
                         df["wlan.fc.type_subtype"]).add_prefix("sub_")
    w = w.join(counts)

    # label of a second = most common attack label in it, otherwise Normal
    attack = df[df["Label"] != "Normal"]
    lab = attack.groupby("frame.time_epoch")["Label"].agg(
        lambda s: s.value_counts().idxmax())
    w["label"] = lab.reindex(w.index).fillna("Normal")
    w["source"] = os.path.basename(path).replace(".csv", "")
    return w.reset_index()


if __name__ == "__main__":
    all_w = pd.concat([build_windows(os.path.join(DATA_DIR, f)) for f in FILES],
                      ignore_index=True)
    sub_cols = [c for c in all_w.columns if c.startswith("sub_")]
    all_w[sub_cols] = all_w[sub_cols].fillna(0)
    all_w[["std_len", "std_rssi"]] = all_w[["std_len", "std_rssi"]].fillna(0)
    all_w["mean_rssi"] = all_w["mean_rssi"].fillna(all_w["mean_rssi"].median())

    out = os.path.join(DATA_DIR, "windows_all.csv")
    all_w.to_csv(out, index=False)
    print("Saved", out, "-", len(all_w), "windows")
    print(pd.crosstab(all_w["source"], all_w["label"]))
