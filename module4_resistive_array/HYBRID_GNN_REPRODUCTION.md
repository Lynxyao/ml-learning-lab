# Hybrid Physics-Correction GNN: PyCharm Reproduction Guide

## Scope

This experiment reconstructs an 8 x 8 local resistance map from 64 pairwise
row-column terminal currents. It is a simulator-only study. Agreement with
Simon's CSV files confirms software consistency with the simulator, not
experimental validation of the physical hardware.

The curriculum generator explicitly excludes every exact resistance-map pattern
present in `R_test.csv`. This prevents exact map duplicates between curriculum
training data and the held-out test set.

## PyCharm setup

Use this project root as the working directory:

```text
C:\Users\10131\PycharmProjects\PythonProject7
```

Select the existing project interpreter and run commands from the PyCharm
Terminal. All paths below are relative to the project root.

## 1. Quick smoke test

```powershell
.\.venv\Scripts\python.exe module4_resistive_array\train_hybrid_physics_gnn.py --data_dir data\resistance_8x8_simon_v2 --output_dir resistance_results\hybrid_smoke --max_train_samples 128 --synthetic_sparse_samples 64 --epochs 2 --batch_size 64 --hidden_size 32 --message_steps 2 --forward_batch_size 8

.\.venv\Scripts\python.exe module4_resistive_array\test_hybrid_physics_gnn.py --data_dir data\resistance_8x8_simon_v2 --checkpoint resistance_results\hybrid_smoke\hybrid_physics_gnn_best.pt --output_dir resistance_results\hybrid_smoke
```

This only verifies that training, the differentiable circuit solver, checkpoint
loading, testing, and plotting work. Do not use its metrics in the meeting.

## 2. Reproduce the main hybrid result

```powershell
.\.venv\Scripts\python.exe module4_resistive_array\train_hybrid_physics_gnn.py --data_dir data\resistance_8x8_simon_v2 --output_dir resistance_results\hybrid_gnn_8x8_v2 --max_train_samples 3000 --synthetic_sparse_samples 3000 --synthetic_max_high_cells 8 --epochs 30 --batch_size 256 --hidden_size 64 --message_steps 3 --forward_batch_size 16 --forward_consistency_weight 0.05

.\.venv\Scripts\python.exe module4_resistive_array\test_hybrid_physics_gnn.py --data_dir data\resistance_8x8_simon_v2 --checkpoint resistance_results\hybrid_gnn_8x8_v2\hybrid_physics_gnn_best.pt --output_dir resistance_results\hybrid_gnn_8x8_v2
```

Seed 7 preliminary results obtained on CPU:

```text
MAE:                         0.099 ohm
RMSE:                        0.552 ohm
High-resistance recall:      1.000
High-resistance precision:   1.000
Exact-map accuracy:          1.000
Log-current RMSE:            0.0196
Uncertainty/error corr.:     0.996
```

Small numerical differences are possible across PyTorch versions.

## 3. Run the no-curriculum ablation

```powershell
.\.venv\Scripts\python.exe module4_resistive_array\train_hybrid_physics_gnn.py --data_dir data\resistance_8x8_simon_v2 --output_dir resistance_results\hybrid_gnn_8x8_v2_no_curriculum --max_train_samples 3000 --synthetic_sparse_samples 0 --epochs 20 --batch_size 256 --hidden_size 64 --message_steps 3 --forward_batch_size 16 --forward_consistency_weight 0.05

.\.venv\Scripts\python.exe module4_resistive_array\test_hybrid_physics_gnn.py --data_dir data\resistance_8x8_simon_v2 --checkpoint resistance_results\hybrid_gnn_8x8_v2_no_curriculum\hybrid_physics_gnn_best.pt --output_dir resistance_results\hybrid_gnn_8x8_v2_no_curriculum
```

Expected preliminary result:

```text
MAE:                    about 3.00 ohm
High-resistance recall: 0.00
```

This is the key evidence that architecture alone does not solve the
train/test-distribution mismatch.

## 4. Prepare a shared curriculum for fair baselines

```powershell
.\.venv\Scripts\python.exe module4_resistive_array\prepare_sparse_curriculum_csv.py --source_dir data\resistance_8x8_simon_v2 --output_dir data\resistance_8x8_simon_v2_curriculum --original_samples 3000 --synthetic_samples 3000 --max_high_cells 8
```

MLP classification baseline:

```powershell
.\.venv\Scripts\python.exe module4_resistive_array\train_real_csv_torch.py --data_dir data\resistance_8x8_simon_v2_curriculum --output_dir resistance_results\fair_curriculum_mlp_classification --model mlp --task classification --positive_state high --threshold 50 --epochs 50 --batch_size 256 --hidden_size 128

.\.venv\Scripts\python.exe module4_resistive_array\test_real_csv_torch.py --data_dir data\resistance_8x8_simon_v2_curriculum --output_dir resistance_results\fair_curriculum_mlp_classification --checkpoint resistance_results\fair_curriculum_mlp_classification\checkpoints\classification_high_mlp_model.pt
```

Grid GNN classification baseline:

```powershell
.\.venv\Scripts\python.exe module4_resistive_array\train_real_csv_torch.py --data_dir data\resistance_8x8_simon_v2_curriculum --output_dir resistance_results\fair_curriculum_grid_gnn_classification --model grid_gnn --task classification --positive_state high --threshold 50 --epochs 30 --batch_size 256 --hidden_size 64

.\.venv\Scripts\python.exe module4_resistive_array\test_real_csv_torch.py --data_dir data\resistance_8x8_simon_v2_curriculum --output_dir resistance_results\fair_curriculum_grid_gnn_classification --checkpoint resistance_results\fair_curriculum_grid_gnn_classification\checkpoints\classification_high_grid_gnn_model.pt
```

## 5. Generate meeting figures and notes

After all result folders exist:

```powershell
.\.venv\Scripts\python.exe module4_resistive_array\make_hybrid_results_presentation.py
```

Main outputs:

```text
resistance_results\hybrid_presentation\model_comparison.png
resistance_results\hybrid_presentation\model_comparison.csv
resistance_results\hybrid_presentation\meeting_results_notes.md
resistance_results\hybrid_gnn_8x8_v2\hybrid_test_examples.png
resistance_results\hybrid_gnn_8x8_v2\hybrid_test_metrics.json
```

## Suggested meeting structure

1. Measurement limitation: show the Jacobian sensitivity map and explain that
   high-resistance cells are roughly 69 times less sensitive in the example.
2. Hybrid design: show `current -> equivalent-resistance proxy -> row/column
   circuit GNN -> local correction -> KCL/Laplacian consistency`.
3. Fair results: compare original models, curriculum baselines, and hybrid.
4. Limitations: all data still comes from the same ideal simulator family.
5. Next step: repeat seeds, add noise and model mismatch, then calibrate with a
   small real dataset.

## Claims to avoid

Do not say that the forward model has been physically validated, that the
hybrid model proves unique recovery in hardware, or that the result is a fully
validated PINN. The accurate description is:

> A physics-proxy, topology-aware hybrid GNN achieved strong reconstruction on
> a preliminary simulator-only sparse-state test, while Jacobian analysis and
> no-curriculum ablation show that identifiability and training-distribution
> design remain central limitations.
