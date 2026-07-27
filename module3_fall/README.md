# Module 3: Fall Prediction With Time-Series RNNs

This module uses the UniMiB-SHAR public accelerometer dataset to teach fall prediction from short motion windows.

## Why This Is Module 3 / RNN

The input is not a single image. Each example is a 3-axis accelerometer sequence with 151 time steps:

- x acceleration over time
- y acceleration over time
- z acceleration over time

The prediction target is binary:

- `0`: activity of daily living
- `1`: fall

This fits an RNN-style module because the model should learn temporal patterns such as a sudden impact followed by reduced movement.

## Files

- `prepare_unimib.py`: converts UniMiB-SHAR `.mat` files into one `.npz` file.
- `fall_dataset.py`: shared dataset loading, splitting, scaling, and metrics helpers.
- `fall_models.py`: GRU and LSTM classifiers.
- `fall_visualize.py`: plots accelerometer examples.
- `fall_train_torch.py`: trains a fall prediction RNN and saves milestone checkpoints.
- `fall_test_torch.py`: evaluates a saved model checkpoint.
- `render_checkpoint_gallery.py`: renders held-out checkpoint comparisons for the website.
- `holomotion_sequence_adapter.py`: converts approved, labeled CSV/JSON motion sequences into the shared NPZ format.
- `example_holomotion_sequence.csv`: minimal long-form example for testing the adapter.

## Quick Start

```powershell
.\.venv313\Scripts\python.exe module3_fall\prepare_unimib.py
.\.venv313\Scripts\python.exe module3_fall\fall_visualize.py
.\.venv313\Scripts\python.exe module3_fall\fall_train_torch.py `
  --epochs 30 `
  --model gru `
  --checkpoint_epochs 5,10,15,20,25,30 `
  --output_dir fall_results\checkpoint_series
.\.venv313\Scripts\python.exe module3_fall\render_checkpoint_gallery.py `
  --results_dir fall_results\checkpoint_series `
  --website_assets website\assets\module3
```

The six checkpoints come from one deterministic run and use the same
subject-held-out split. They are intended for learning-curve inspection. Formal
model selection should use validation performance rather than repeatedly choosing
the best test-set result.

## Teaching Focus

Students should compare:

- Why accuracy alone is not enough for fall prediction.
- Why recall/sensitivity matters when the positive class is a fall.
- What a fall-like acceleration trace looks like compared with daily activity.
- How training/test split choices affect real-world reliability.

## Holomotion Data Boundary

The current saved GRU was trained on UniMiB-SHAR smartphone accelerometer windows,
not on Holomotion data. Holomotion exports may contain joint coordinates, joint
angles, gait timing, or other biomechanical fields, so they are a different feature
space and require a newly trained checkpoint.

For a labeled time-series export, use long-form CSV or JSON rows with:

```text
sample_id, frame, label, subject_id, <numeric feature 1>, <numeric feature 2>, ...
```

Each `sample_id` must contain at least two frames and one consistent binary label.
The adapter resamples each sequence to a fixed length while preserving every numeric
feature column:

```powershell
.\.venv313\Scripts\python.exe module3_fall\holomotion_sequence_adapter.py `
  --input approved_holomotion_sequences.csv `
  --output data\fall\holomotion_labeled_windows.npz `
  --sequence_length 151

.\.venv313\Scripts\python.exe module3_fall\fall_train_torch.py `
  --data_npz data\fall\holomotion_labeled_windows.npz `
  --epochs 30 `
  --checkpoint_epochs 5,10,15,20,25,30 `
  --output_dir fall_results\holomotion_sequence_run
```

Supervised fall-risk training requires labels such as observed falls, a prespecified
clinical outcome, or an approved reference annotation. An unlabeled device report
can be inspected or clustered, but it cannot validate a supervised fall classifier.
Human-subject motion data should remain local unless an approved consent, storage,
and access-control plan permits upload.

## Public Dataset

The dataset comes from UniMiB-SHAR:

Micucci, D., Mobilio, M., & Napoletano, P. (2017). UniMiB SHAR: A new dataset for human activity recognition using acceleration data from smartphones.
