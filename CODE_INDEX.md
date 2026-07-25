# Code Index

This repository pairs a static student website with local prototype code.

## Website

- `website/index.html` - static interactive learning site
- `website/styles.css` - site styles
- `website/app.js` - browser interactions, quizzes, simulations, and local-backend detection
- `website/assets/` - small public figures and reference assets used by the website

## Module Code

- `module1_wfm/` - WFM image-to-image learning prototype
- `module2_ecg/` - ECG beat classification prototype
- `module3_fall/` - public fall/motion sequence modeling prototype
- `module4_resistive_array/` - 8x8 resistive-array inverse-modeling research prototype
  - `run_mlp_baseline_8x8.py` - data-only MLP baseline
  - `hybrid_physics_gnn.py` - V/I proxy plus local GNN correction and uncertainty
  - `train_hybrid_physics_gnn.py` / `test_hybrid_physics_gnn.py` - Hybrid GNN workflow
  - `dynamic_impedance/` - frequency-ready circuit graph, independent MNA, losses, and tests
  - `train_dynamic_impedance_dc.py` / `test_dynamic_impedance_dc.py` - present DC MNA-GNN workflow
  - `analyze_pairwise_jacobian.py` - local sensitivity and identifiability diagnostics
- `backend_server.py` - local-only teaching backend for realtime demos

## Not Included in GitHub

The repository intentionally excludes large or local-only files through `.gitignore`, including:

- virtual environments
- large datasets
- PyTorch checkpoints
- generated experiment outputs
- local run folders

The GitHub Pages site should be treated as the public static version. Realtime training remains a local instructor/developer mode.
