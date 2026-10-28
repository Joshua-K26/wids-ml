"""Step 3: the detector. Replays a capture second by second, raises alerts, logs incidents.

The model is trained WITHOUT the replayed capture, so the alerts are honest.
Run   : python src/detector.py                       (replays Disasso.csv)
        python src/detector.py --delay 0.1           (slower, nicer for a demo)
        python src/detector.py --retrain             (ignore the saved model, train again)
Times are shown in UTC (the capture's own clock) unless --tz-offset is given.
"""
import argparse
import os
from collections import deque
from datetime import datetime, timezone, timedelta
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from wids_common import (OUT_DIR, MODEL_DIR, CLASSES, CLASS_ID, COMPACT, SEQ_LEN, AUG_COPIES,
                         Net, load_windows, seqs_from, augmented_copy, train_net)

parser = argparse.ArgumentParser()
parser.add_argument("--demo", default="Disasso", help="capture to replay (excluded from training)")
parser.add_argument("--gap", type=int, default=3, help="calm seconds that end an incident")
parser.add_argument("--delay", type=float, default=0.0, help="seconds to pause between windows")
parser.add_argument("--tz-offset", type=float, default=0.0, help="hours added to UTC for display")
parser.add_argument("--retrain", action="store_true")
args = parser.parse_args()

w = load_windows()
MODEL_PATH = os.path.join(MODEL_DIR, "wids_lstm.pt")

# ---------- 1. Get the trained model ----------
if os.path.exists(MODEL_PATH) and not args.retrain:
    # weights_only=False because the file also stores numpy scaling numbers (our own file)
    saved = torch.load(MODEL_PATH, weights_only=False)
    net = Net(len(saved["features"]))
    net.load_state_dict(saved["state"])
    net.eval()
    mu, sd = saved["mu"], saved["sd"]
    print("Loaded saved model:", MODEL_PATH)
else:
    print("Training a model without", args.demo, "...")
    rng = np.random.default_rng(1)
    torch.manual_seed(1)
    y_all = w["label3"].map(CLASS_ID).values
    tr = (w["source"] != args.demo).values
    copies = [augmented_copy(w, rng) for _ in range(AUG_COPIES)]
    Xtr = np.concatenate([seqs_from(w, COMPACT)[tr]] + [seqs_from(a, COMPACT)[tr] for a in copies])
    ytr = np.concatenate([y_all[tr]] * (AUG_COPIES + 1))
    net, mu, sd = train_net(Xtr, ytr)
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "features": COMPACT,
                "classes": CLASSES, "L": SEQ_LEN}, MODEL_PATH)
    print("Saved model:", MODEL_PATH)


def classify(buffer):
    """Verdict for the newest second, using the last 10 seconds in the buffer."""
    x = np.log1p(np.clip(np.array(buffer, dtype="float32"), 0, None))
    x = torch.tensor(((x - mu) / sd)[None], dtype=torch.float32)
    with torch.no_grad():
        return CLASSES[net(x).argmax(1).item()]


# ---------- 2. Replay the capture like a live feed ----------
stream = w[w["source"] == args.demo].sort_values("frame.time_epoch").reset_index(drop=True)
tz = timezone(timedelta(hours=args.tz_offset))
log_path = os.path.join(OUT_DIR, "alerts.log")
open(log_path, "w").close()


def log(msg):
    print(msg)
    with open(log_path, "a") as f:
        f.write(msg + "\n")


buffer = deque(maxlen=SEQ_LEN)
in_attack, calm, current = False, 0, None
incidents, preds = [], []

for _, row in stream.iterrows():
    raw = row[COMPACT].values.astype("float32")
    if not buffer:
        for _ in range(SEQ_LEN):          # fill the buffer at the start of the capture
            buffer.append(raw)
    else:
        buffer.append(raw)
    verdict = classify(buffer)
    preds.append(verdict)
    t = datetime.fromtimestamp(row["frame.time_epoch"], tz=tz).strftime("%H:%M:%S")

    if verdict != "Normal":
        calm = 0
        if not in_attack:                  # first attack second = raise the alert
            in_attack = True
            current = {"start": t, "end": t, "seconds": 1, "type": verdict}
            incidents.append(current)
            log(f"[ALERT {t}] {verdict} detected "
                f"(disconnect frames/sec: {int(row['disc'])}, beacons/sec: {int(row['sub_8'])})")
            log(f"[ACTION {t}] Incident logged; channel flagged for investigation")
        else:
            current["end"] = t
            current["seconds"] += 1
    elif in_attack:
        calm += 1
        if calm >= args.gap:               # enough calm seconds = incident over
            in_attack = False
            log(f"[CLEAR {t}] Traffic back to normal")
    if args.delay:
        time.sleep(args.delay)

# ---------- 3. Summary ----------
truth = stream["label3"].values
preds = np.array(preds)
attack = truth != "Normal"
log("\n--- SUMMARY (replayed capture the model never saw) ---")
log(f"Seconds scanned: {len(stream)}")
log(f"True attack seconds: {attack.sum()} | caught: {((preds != 'Normal') & attack).sum()}")
log(f"False-alarm seconds: {((preds != 'Normal') & ~attack).sum()}")
log(f"Incidents raised: {len(incidents)}")

# ---------- 4. Timeline picture ----------
fig, ax = plt.subplots(figsize=(12, 3.5))
x = np.arange(len(stream))
top = stream["disc"].max()
ax.plot(x, stream["disc"], lw=0.8, color="#444")
ax.fill_between(x, 0, top, where=attack, color="red", alpha=0.15, label="True attack")
flag = preds != "Normal"
ax.scatter(x[flag], np.full(flag.sum(), top * 1.03), s=6, color="red", label="Detector flagged")
ax.set_xlabel("Seconds into capture")
ax.set_ylabel("Disconnect frames / sec")
ax.set_title(f"LSTM detector on unseen capture ({args.demo}.csv)")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "detector_timeline.png"), dpi=200)
