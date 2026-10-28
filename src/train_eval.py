"""Step 2: train and evaluate the models (Random Forest baseline and LSTM).

Parts:
  1. Block cross-validation (test blocks come from time ranges the model did not train on)
  2. Unseen-capture test (train on other captures, test on a whole capture never seen),
     without and with data augmentation
Run   : python src/train_eval.py            (5 seeds, takes a while)
        python src/train_eval.py --seeds 1  (quick check)
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)
from wids_common import (OUT_DIR, CLASSES, CLASS_ID, COMPACT, GENERAL, AUG_COPIES,
                         load_windows, sub_columns, seqs_from, augmented_copy, fit_predict)

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=5)
args = parser.parse_args()

w = load_windows()
y_all = w["label3"].map(CLASS_ID).values
X_seq = seqs_from(w, COMPACT)
FULL = GENERAL + sub_columns(w)
report_file = open(os.path.join(OUT_DIR, "metrics.txt"), "w")


def say(text=""):
    print(text)
    report_file.write(str(text) + "\n")


def rf():
    return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=1)


def show_result(title, truth, pred, png):
    say("=" * 60)
    say(title)
    say(f"Accuracy: {accuracy_score(truth, pred):.3f} | "
        f"Macro F1: {f1_score(truth, pred, average='macro'):.3f}")
    say(classification_report(truth, pred, labels=CLASSES, zero_division=0))
    cm = confusion_matrix(truth, pred, labels=CLASSES)
    say(pd.DataFrame(cm, index=CLASSES, columns=CLASSES))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Blues", xticks_rotation=20)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, png), dpi=200)
    plt.close()


# ---------- Part 1: block cross-validation ----------
say("Lazy baseline (always 'Normal'): %.3f" % (w["label3"] == "Normal").mean())
folds = list(GroupKFold(n_splits=5).split(w, w["label3"], w["block"]))

for name, feats in {"Full": FULL, "Blind": GENERAL, "Compact": COMPACT}.items():
    pred = np.empty(len(w), dtype=object)
    for tr, te in folds:
        pred[te] = rf().fit(w.loc[tr, feats], w.loc[tr, "label3"]).predict(w.loc[te, feats])
    show_result(f"Random Forest, {name} features, block CV",
                w["label3"].values, pred, f"rf3_confusion_{name}.png")

torch.manual_seed(1)
pred = np.zeros(len(w), dtype=int)
for tr, te in folds:
    pred[te] = fit_predict(X_seq[tr], y_all[tr], X_seq[te])
show_result("LSTM, Compact features, block CV",
            w["label3"].values, np.array(CLASSES)[pred], "lstm_confusion.png")


# ---------- Part 2: unseen-capture test ----------
def unseen_capture(held, augment, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    tr = (w["source"] != held).values
    te = ~tr
    Xtab, ytab = w.loc[tr, COMPACT], w.loc[tr, "label3"]
    Xs, ys = X_seq[tr], y_all[tr]
    if augment:
        copies = [augmented_copy(w, rng) for _ in range(AUG_COPIES)]
        Xtab = pd.concat([Xtab] + [a.loc[tr, COMPACT] for a in copies])
        ytab = pd.concat([ytab] * (AUG_COPIES + 1))
        Xs = np.concatenate([Xs] + [seqs_from(a, COMPACT)[tr] for a in copies])
        ys = np.concatenate([ys] * (AUG_COPIES + 1))
    truth = w.loc[te, "label3"].values
    attack = truth == "Disconnect"
    out = {}
    p_rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                  random_state=seed).fit(Xtab, ytab).predict(w.loc[te, COMPACT])
    p_lstm = np.array(CLASSES)[fit_predict(Xs, ys, X_seq[te])]
    for model, p in {"RF": p_rf, "LSTM": p_lstm}.items():
        recall = ((p == truth) & attack).sum() / attack.sum()
        false_alarms = ((p != "Normal") & (truth == "Normal")).sum()
        out[model] = (recall, false_alarms)
    return out


say("=" * 60)
say("Unseen-capture test (train without the capture, then test on it)")
for augment in [False, True]:
    for held in ["Deauth", "Disasso"]:
        runs = [unseen_capture(held, augment, s) for s in range(1, args.seeds + 1)]
        for model in ["RF", "LSTM"]:
            rec = [r[model][0] for r in runs]
            fa = [r[model][1] for r in runs]
            say(f"augmentation={str(augment):5s} held out {held:8s} {model:4s} "
                f"attack recall {np.mean(rec):.3f} +/- {np.std(rec):.3f} | "
                f"false alarms {np.mean(fa):.1f}")
report_file.close()
