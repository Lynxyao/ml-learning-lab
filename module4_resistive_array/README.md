# Module 4: Resistive Array Inverse Sensing

This module treats resistive-array reconstruction as an inverse sensing problem.

## Current Public Snapshot

Updated 2026-07-27. The GitHub version now includes:

- the data-only 8x8 MLP baseline;
- the Hybrid GNN with a current-derived proxy, local correction, and uncertainty;
- the independent DC/AC modified nodal analysis backend;
- Jacobian sensitivity and identifiability diagnostics;
- frequency-ready resistance, inductance, and capacitance prediction heads; and
- unit tests for Ohm's law, series/parallel circuits, DC/AC MNA, Jacobians, model
  output masks, and differentiable physics loss.

Run the eight circuit/model tests from this module directory:

```powershell
.\.venv313\Scripts\python.exe -m unittest discover -s dynamic_impedance\tests -v
```

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
5. Optionally add a simulator-informed consistency loss so predicted maps reproduce the measured signals under an unvalidated ideal circuit model.
6. Use Jacobian sensitivity analysis to check whether the measurement design can distinguish local cells before scaling to higher-order arrays.

Approved experimental measurements can replace the synthetic generator later.

## Files

- `resistance_physics.py`: forward simulator from hidden local map to measurable signals.
- `generate_synthetic_resistance.py`: creates a synthetic `.npz` dataset.
- `resistance_dataset.py`: loading, splitting, and scaling helpers.
- `resistance_models.py`: linear and MLP inverse models.
- `train_inverse_torch.py`: trains the inverse reconstruction model.
- `test_inverse_torch.py`: evaluates a checkpoint and exports visual results.
- `train_real_csv_torch.py`: trains the current real-CSV model on `I_train.csv` and `R_train.csv`.
- `test_real_csv_torch.py`: evaluates the saved real-CSV checkpoint on `I_test.csv` and `R_test.csv`.
- `physics_torch.py`: differentiable forward model used for physics-informed training.
- `analyze_forward_jacobian.py`: reports measurement sensitivity and local identifiability diagnostics.
- `circuit_physics.py`: ideal KCL/Laplacian solver for row-column terminal-pair measurements.
- `analyze_pairwise_jacobian.py`: computes the full n^2 by n^2 Jacobian and singular-value diagnostics for pairwise terminal-current measurements.
- `hybrid_physics_gnn.py`: physics-proxy correction GNN with row/column circuit message passing, uncertainty, and calibration parameters.
- `train_hybrid_physics_gnn.py`: trains the hybrid model with optional sparse-state simulator curriculum.
- `test_hybrid_physics_gnn.py`: evaluates resistance, high-state, current-consistency, and uncertainty metrics.
- `prepare_sparse_curriculum_csv.py`: creates a shared curriculum dataset for fair MLP/GNN comparisons.
- `make_hybrid_results_presentation.py`: builds the comparison table, figure, and meeting notes.
- `analyze_sensitivity.py`: explains why low-resistance regions are easier to recover than high-resistance regions.
- `dynamic_impedance/`: frequency-ready circuit graph, independent DC/AC MNA, RLC GNN, physics losses, and tests.
- `train_dynamic_impedance_dc.py`: trains the new architecture on the existing real-valued DC CSV format.
- `test_dynamic_impedance_dc.py`: exports DC metrics, predictions, uncertainty, and heatmaps.
- `analyze_dynamic_impedance_jacobian.py`: shared DC/AC parameter-sensitivity analysis.

## Quick Start

```powershell
.\.venv313\Scripts\python.exe module4_resistive_array\generate_synthetic_resistance.py --samples 5000 --grid_size 3 --measurement_noise 0.02
.\.venv313\Scripts\python.exe module4_resistive_array\train_inverse_torch.py --epochs 80 --model mlp
.\.venv313\Scripts\python.exe module4_resistive_array\test_inverse_torch.py --checkpoint resistance_results\checkpoints\best_inverse_model.pt
.\.venv313\Scripts\python.exe module4_resistive_array\analyze_sensitivity.py
```

Simulator-consistency 3x3 experiment:

```powershell
.\.venv313\Scripts\python.exe module4_resistive_array\train_inverse_torch.py --epochs 80 --model mlp --physics_loss_weight 0.1
```

Forward-model Jacobian checks:

```powershell
.\.venv313\Scripts\python.exe module4_resistive_array\analyze_forward_jacobian.py --grid_size 3
.\.venv313\Scripts\python.exe module4_resistive_array\analyze_forward_jacobian.py --grid_size 7
.\.venv313\Scripts\python.exe module4_resistive_array\analyze_pairwise_jacobian.py --data_dir data\resistance_8x8_simon_v2
```

## Real CSV Train/Test

```powershell
.\.venv313\Scripts\python.exe module4_resistive_array\train_real_csv_torch.py
.\.venv313\Scripts\python.exe module4_resistive_array\train_real_csv_torch.py --forward_consistency_weight 0.02
.\.venv313\Scripts\python.exe module4_resistive_array\test_real_csv_torch.py --checkpoint resistance_results\real_csv\checkpoints\regression_conductance_mlp_forward_w0p02_model.pt
```

## Hybrid Physics-Correction GNN

```powershell
.\.venv313\Scripts\python.exe module4_resistive_array\train_hybrid_physics_gnn.py --data_dir data\resistance_8x8_simon_v2 --output_dir resistance_results\hybrid_gnn_8x8_v2 --max_train_samples 3000 --synthetic_sparse_samples 3000 --synthetic_max_high_cells 8 --epochs 30 --batch_size 256 --hidden_size 64 --message_steps 3 --forward_batch_size 16 --forward_consistency_weight 0.05
.\.venv313\Scripts\python.exe module4_resistive_array\test_hybrid_physics_gnn.py --data_dir data\resistance_8x8_simon_v2 --checkpoint resistance_results\hybrid_gnn_8x8_v2\hybrid_physics_gnn_best.pt --output_dir resistance_results\hybrid_gnn_8x8_v2
.\.venv313\Scripts\python.exe module4_resistive_array\make_hybrid_results_presentation.py
```

The sparse curriculum and test maps currently come from the same ideal
simulator family. Report these as preliminary simulator-only results, not as
experimental validation.

## Frequency-Ready Dynamic Impedance Model

The new V2 architecture preserves the current DC workflow while preparing for
complex frequency-domain measurements. DC mode trains only resistance. The L
and C heads are present but masked and explicitly recorded as untrained until
frequency and phase data become available. See
`dynamic_impedance/README.md` for commands and scope limitations.

## Outputs

- `resistance_results/test_metrics.json`
- `resistance_results/test_predictions.npz`
- `resistance_results/test_sample_predictions.csv`
- `resistance_results/figures/resistance_inverse_examples.png`
- `resistance_results/sensitivity/low_vs_high_current_sensitivity.png`
- `resistance_results/jacobian/jacobian_summary_*x*.json`
- `resistance_results/pairwise_jacobian_8x8/jacobian_summary.csv`
- `resistance_results/pairwise_jacobian_8x8/forward_model_data_agreement.json`

The data-agreement check only verifies that the Python circuit solver reproduces
the reference-model-generated CSV currents. It is not experimental validation of the circuit
topology, floating-terminal assumptions, contact resistance, or parasitic effects.

## Research Extension

The next research step is not simply to improve model accuracy. The important questions are:

- Which measurement design is sufficient to recover local biological patterns?
- How much reconstruction quality is lost when the number of measured paths is reduced?
- Can a model trained on simulated circuit data transfer to experimental data?
- Which local regions are most uncertain, and should those regions trigger additional measurements?
- Can reconstruction error identify abnormal cell growth or sensor failure?
