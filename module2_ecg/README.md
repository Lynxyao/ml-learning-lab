# Module 2: ECG Time-Series Classification

This folder is an independent scaffold for the ECG signal module. It does not modify the existing WFM image-to-image files.

## Goal

Train a time-series model on segmented ECG beats and classify arrhythmia beat categories. The default target dataset is MIT-BIH Arrhythmia Database, but the training code reads a simple prepared `.npz` file so other ECG datasets can be plugged in later.

## Files

- `ecg_network_torch.py`: 1D CNN, LSTM, and CNN-LSTM models.
- `download_mitdb.py`: download MIT-BIH records from PhysioNet with `wfdb`.
- `ecg_dataset.py`: dataset classes, label mapping, and optional MIT-BIH preprocessing helper.
- `ecg_train_torch.py`: train ECG classifiers.
- `ecg_test_torch.py`: evaluate a trained checkpoint and save metrics.
- `ecg_visualize.py`: save ECG segment plots for teaching/demo pages.
- `generate_demo_ecg_data.py`: create synthetic ECG-like data for quick smoke tests.
- `requirements_ecg.txt`: optional ECG-specific dependencies.

## Expected Prepared Data Format

The training scripts expect:

```text
data/ecg/mitdb_beat_segments.npz
```

with arrays:

```python
signals: float32 array, shape [N, L] or [N, C, L]
labels: int64 array, shape [N]
label_names: optional string array, shape [num_classes]
```

For a first module, use five AAMI-style classes:

```text
0 = N     normal and bundle branch block beats
1 = SVEB  supraventricular ectopic beats
2 = VEB   ventricular ectopic beats
3 = F     fusion beats
4 = Q     unknown / paced / unclassifiable beats
```

## Quick Start

Recommended PyCharm interpreter after setup:

```text
C:\Users\10131\PycharmProjects\PythonProject5\.venv310\Scripts\python.exe
```

### Option A: Smoke Test With Synthetic Data

This checks that the module code runs before using real PhysioNet data.

```bash
python module2_ecg/generate_demo_ecg_data.py
python module2_ecg/ecg_train_torch.py --data_npz data/ecg/demo_beat_segments.npz --model cnn --epochs 5
python module2_ecg/ecg_test_torch.py --data_npz data/ecg/demo_beat_segments.npz --checkpoint ecg_results/checkpoints/best_model.pth --model cnn
python module2_ecg/ecg_visualize.py --data_npz data/ecg/demo_beat_segments.npz --max_samples 12
```

### Option B: Train With Prepared MIT-BIH Data

Train a small 1D CNN:

```bash
python module2_ecg/ecg_train_torch.py --data_npz data/ecg/mitdb_beat_segments.npz --model cnn
```

Evaluate:

```bash
python module2_ecg/ecg_test_torch.py --data_npz data/ecg/mitdb_beat_segments.npz --checkpoint ecg_results/checkpoints/best_model.pth --model cnn
```

Visualize samples:

```bash
python module2_ecg/ecg_visualize.py --data_npz data/ecg/mitdb_beat_segments.npz --max_samples 12
```

## Preparing MIT-BIH Data

If you install `wfdb`, the dataset helper can convert local MIT-BIH records into beat segments:

```bash
pip install -r module2_ecg/requirements_ecg.txt
python module2_ecg/ecg_dataset.py --mitdb_dir data/ecg/mitdb --out_npz data/ecg/mitdb_beat_segments.npz
```

The refined training default uses a record-wise split to reduce leakage between train,
validation, and test sets:

```bash
python module2_ecg/ecg_train_torch.py --data_npz data/ecg/mitdb_beat_segments.npz --model cnn --split_mode record --class_weighting inverse
python module2_ecg/ecg_test_torch.py --data_npz data/ecg/mitdb_beat_segments.npz --checkpoint ecg_results/checkpoints/best_model.pth --model cnn --split_mode record
```

For teaching demos, compare `accuracy` with `macro_f1` in `test_metrics.json`.
Accuracy can look high on MIT-BIH because normal beats dominate the dataset.

Download MIT-BIH from PhysioNet first and place records under `data/ecg/mitdb`.

Or download with `wfdb`:

```bash
python module2_ecg/download_mitdb.py --out_dir data/ecg/mitdb
python module2_ecg/ecg_dataset.py --mitdb_dir data/ecg/mitdb --out_npz data/ecg/mitdb_beat_segments.npz
```
