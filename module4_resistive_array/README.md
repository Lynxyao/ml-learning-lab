# Module 4: Resistive Array Inverse Sensing

This module prototypes Simon's resistive-array idea as an inverse sensing problem.

## Core Problem

In the real system, each node or local region in an `n x n` resistive array may change resistance as cells attach, spread, or alter the local environment. Directly measuring every local node in real time may be difficult or expensive. The easier signals are aggregate measurements such as row, column, path, or whole-array equivalent resistance.

The ML task is:

```text
measurable row/column/path signals -> hidden local resistance map
```

This is deeper than a simple classification task because the model must reconstruct a spatial state that is only indirectly observed.

## Current Prototype

The current code uses a teaching simulator:

1. Generate synthetic local resistance maps with smooth cell-growth-like patterns.
2. Convert each hidden map into fixed-voltage current measurements from rows, columns, diagonals, and the whole array.
3. Train an inverse model to reconstruct the hidden map.
4. Evaluate the prediction with MAE, RMSE, pattern correlation, heatmaps, and CSV outputs.

Simon or Esther's real data can replace the synthetic generator later.

## Files

- `resistance_physics.py`: forward simulator from hidden local map to measurable signals.
- `generate_synthetic_resistance.py`: creates a synthetic `.npz` dataset.
- `resistance_dataset.py`: loading, splitting, and scaling helpers.
- `resistance_models.py`: linear and MLP inverse models.
- `train_inverse_torch.py`: trains the inverse reconstruction model.
- `test_inverse_torch.py`: evaluates a checkpoint and exports visual results.
- `train_real_csv_torch.py`: trains the current real-CSV model on `I_train.csv` and `R_train.csv`.
- `test_real_csv_torch.py`: evaluates the saved real-CSV checkpoint on `I_test.csv` and `R_test.csv`.
- `analyze_sensitivity.py`: explains why low-resistance regions are easier to recover than high-resistance regions.

## Quick Start

```powershell
.\.venv313\Scripts\python.exe module4_resistive_array\generate_synthetic_resistance.py --samples 5000 --grid_size 3 --measurement_noise 0.02
.\.venv313\Scripts\python.exe module4_resistive_array\train_inverse_torch.py --epochs 80 --model mlp
.\.venv313\Scripts\python.exe module4_resistive_array\test_inverse_torch.py --checkpoint resistance_results\checkpoints\best_inverse_model.pt
.\.venv313\Scripts\python.exe module4_resistive_array\analyze_sensitivity.py
```

## Real CSV Train/Test

```powershell
.\.venv313\Scripts\python.exe module4_resistive_array\train_real_csv_torch.py
.\.venv313\Scripts\python.exe module4_resistive_array\test_real_csv_torch.py --checkpoint resistance_results\real_csv\checkpoints\regression_conductance_physics_w0p1_model.pt
```

## Outputs

- `resistance_results/test_metrics.json`
- `resistance_results/test_predictions.npz`
- `resistance_results/test_sample_predictions.csv`
- `resistance_results/figures/resistance_inverse_examples.png`
- `resistance_results/sensitivity/low_vs_high_current_sensitivity.png`

## Research Extension

The next research step is not simply to improve model accuracy. The important questions are:

- Which measurement design is sufficient to recover local biological patterns?
- How much reconstruction quality is lost when the number of measured paths is reduced?
- Can a model trained on simulated circuit data transfer to Esther's experimental data?
- Which local regions are most uncertain, and should those regions trigger additional measurements?
- Can reconstruction error identify abnormal cell growth or sensor failure?
