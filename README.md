# wids-ml: Wi-Fi intrusion detection with machine learning

A prototype wireless intrusion detection system (WIDS). It reads Wi-Fi frames, summarises each
second of traffic into a few numbers, and uses machine learning (a Random Forest baseline and an
LSTM) to label every second as **Normal**, **Disconnect flood** (deauthentication or
disassociation) or **Beacon flood**. When an attack is detected it raises an alert and writes it
to a log.

## Data

Public **WPA3 Attacks Dataset** (Mendeley Data, DOI 10.17632/cxx5t5nw7z.1), recorded in a lab
with one WPA3-Personal access point and four stations. Access to these CSV files are public from
<https://data.mendeley.com/datasets/cxx5t5nw7z/1> . I used below the three attack csv files to build my system

- `Deauth.csv` (deauthentication flood)
- `Disasso.csv` (disassociation flood, also contains deauth frames)
- `BeaconFlood.csv` (beacon flood)

The raw files are not stored in this repository (too large, and they belong to the dataset
authors). `data/windows_all.csv` is my own per-second summary computed from them.
No traffic was captured or attacked by me; everything uses this public dataset.

## How it works

1. **Windows.** One frame says little, so frames are grouped into 1-second windows.
2. **Features** per second: frame counts by type, share of disconnect frames, share of beacons,
   total frames, frame size and signal strength.
3. **Models.** A Random Forest (many decision trees voting) judges each second alone. An LSTM
   reads the last 10 seconds, so it can notice "much busier than a moment ago".
4. **Detector.** Replays a capture second by second, raises an alert on the first attack second,
   merges neighbouring attack seconds into one incident, and logs it.

## Run it

**Quick start (no download needed).** The repository includes the small summary table
`data/windows_all.csv` (one row per second, about 2,300 rows) and the trained model
`models/wids_lstm.pt`, so the detector and the evaluation run straight from a fresh clone:

```
git clone https://github.com/Joshua-K26/wids-ml
cd wids-ml
pip install -r requirements.txt      # Python 3.10-3.12 recommended; PyTorch is a large download
python src/detector.py               # replay + alerts, writes results/alerts.log
python src/train_eval.py --seeds 1   # evaluation (use the default 5 seeds for the final numbers)
```

The same commands work in Google Colab (put `!` before each, and run `%cd wids-ml` after the clone).
Optional flags: `python src/detector.py --delay 0.1` replays slowly for a demo, `--retrain` trains a new model.

**Full pipeline from the raw dataset.** To rebuild `windows_all.csv` yourself, download the three
CSVs listed under Data into `data/` and run `python src/make_windows.py` first.

## Results

Plain accuracy is misleading here (always answering "Normal" already scores 60.8%), so the
tables show per-class results. Pictures are in `results/`.

**Same-capture test (block cross-validation, 3 classes)**

| Model | Accuracy | Macro F1 |
|---|---|---|
| Random Forest, Compact features | 99.2% | 0.987 |
| LSTM, Compact features (5 seeds) | 99.7% +/- 0.1 | 0.996 +/- 0.001 |
| Random Forest, Blind features (no frame-type info) | 79.7% | 0.744 |

**Harder test: a whole capture the model never saw.** Attack recall on `Disasso.csv`, which holds
much weaker floods than the training data:

| Model | Without augmentation | With augmentation |
|---|---|---|
| Random Forest | 31.2% | 93.1% +/- 0.2 |
| LSTM (5 seeds) | 47.2% +/- 5.3 | 89.7% +/- 3.8 |

Both models caught 100% of the floods in `Deauth.csv` when it was held out.
Augmentation (training on artificially weakened attacks) mattered more than the choice of model.

**Detector demo** on the unseen `Disasso.csv`: 229 of 240 attack seconds caught, 0 false-alarm
seconds, 20 incidents (the attack comes in bursts). This is one run; the 5-seed averages above are
the better estimate.

## Limitations

- The CSVs contain no MAC addresses, so the system can log and flag attack windows but cannot
  block a device. It is a WIDS with response logging, not a full WIPS. The PCAP files keep
  addresses and would allow a MAC blocklist.
- Deauth and disassociation could not be told apart (the attack tool sent both), so they are
  merged into one class. The four-class result is in `results/rf_confusion_*.png`.
- The captures differ even in normal traffic, so same-capture scores (about 99%) are optimistic.
  The unseen-capture test is the more honest one.
- The augmentation was designed after seeing the weak-attack failure, and weak attacks are
  simulated by shrinking real counts. Only two held-out captures; everything comes from one lab.
- Times in the log are UTC, from the capture's own clock.

## Early approach (not used)

My first version split frames randomly on one capture and scored 100%. That was misleading:
the label was identical to "is this a deauth frame", and neighbouring seconds leaked between
training and test data. This repository replaces it with block-based and unseen-capture testing.

## AI assistance

I used Claude (Anthropic) to help plan the pipeline, write and debug the code, and explain the
machine learning concepts. I ran every experiment myself on the dataset and can explain how each
part works.
