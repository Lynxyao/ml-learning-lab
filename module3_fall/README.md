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
- `fall_train_torch.py`: trains a fall prediction RNN.
- `fall_test_torch.py`: evaluates a saved model checkpoint.

## Quick Start

```powershell
.\.venv313\Scripts\python.exe module3_fall\prepare_unimib.py
.\.venv313\Scripts\python.exe module3_fall\fall_visualize.py
.\.venv313\Scripts\python.exe module3_fall\fall_train_torch.py --epochs 20 --model gru
.\.venv313\Scripts\python.exe module3_fall\fall_test_torch.py --checkpoint fall_results\checkpoints\best_model.pt
```

## Teaching Focus

Students should compare:

- Why accuracy alone is not enough for fall prediction.
- Why recall/sensitivity matters when the positive class is a fall.
- What a fall-like acceleration trace looks like compared with daily activity.
- How training/test split choices affect real-world reliability.

## Public Dataset

The dataset comes from UniMiB-SHAR:

Micucci, D., Mobilio, M., & Napoletano, P. (2017). UniMiB SHAR: A new dataset for human activity recognition using acceleration data from smartphones.
