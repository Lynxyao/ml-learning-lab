# 8 x 8 Plain MLP Baseline

This baseline maps 64 current values directly to 64 conductance targets and
then converts them back to resistance. It uses no GNN, no forward-consistency
loss, no MNA/KCL loss, no Jacobian term, and no uncertainty/classification head.

## Historical baseline

Run `run_mlp_baseline_8x8.py` directly in PyCharm with no program arguments.

- Original samples: 10,000
- Sparse curriculum: 0
- Physics-informed loss weight: 0
- Output: `resistance_results/mlp_baseline_8x8_original`

## Curriculum-controlled baseline

Use this PyCharm program argument:

```text
--curriculum_samples 4000
```

This model is still a plain MLP with no physics-informed loss. The extra maps
only control for the training-distribution improvement used by the two GNNs.
The output is `resistance_results/mlp_baseline_8x8_curriculum4000`.

## Fair interpretation

- Historical MLP vs Hybrid/MNA-GNN: total pipeline improvement.
- Curriculum MLP vs Hybrid/MNA-GNN: improvement beyond sparse-data balancing.
- Hybrid vs MNA-GNN: effect of the two graph/physics formulations.

Use the same seed, test CSV, threshold, and resistance metrics for all models.
