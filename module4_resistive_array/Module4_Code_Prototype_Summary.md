# Module 4 Code Prototype Summary: Resistive Array Inverse Sensing

## What This Prototype Does

This code treats Simon's resistive-array project as an inverse sensing problem. In the real system, local regions of an electrical array may change resistance as cells attach, grow, or change their local state. Simon's current setup can be interpreted as applying a fixed voltage and measuring current. Since current depends on conductance, low-resistance regions dominate the signal while high-resistance regions contribute weakly.

The current prototype asks:

```text
Can a model infer the hidden local resistance map from limited measurable resistance signals?
```

## Current Input and Output

Input:

- measurable current features from an array:
  - row current values
  - column current values
  - diagonal/path current values
  - whole-array current value

Output:

- A reconstructed 4x4 local resistance map.
- Each predicted value corresponds to one hidden local region in the array.

## Why This Is Deeper Than Classification

This is not simply classifying a sample as normal or abnormal. The model is trying to reconstruct a spatial internal state from incomplete external measurements. This makes it an inverse mapping problem:

```text
forward problem: local resistance map -> measurable resistance signals
inverse problem: measurable resistance signals -> local resistance map
```

Because the input has fewer directly measured values than the number of hidden local values, the reconstruction is underdetermined. The model must learn spatial patterns and constraints from data.

## What Was Implemented

- Synthetic resistance-map generator with smooth local patterns that mimic cell-growth-related changes.
- Measurement simulator that converts each hidden map into row, column, diagonal, and whole-array resistance signals.
- PyTorch inverse models:
  - linear baseline
  - MLP inverse model
- Train/test pipeline with saved checkpoint.
- Evaluation outputs:
  - MAE in ohms
  - RMSE in ohms
  - pattern correlation between predicted and true local maps
  - true-vs-predicted heatmap figure
  - CSV table containing test sample predictions

## First Test Result

Using a synthetic array with 2% measurement noise:

- Test MAE: about 30.4 ohms
- Test RMSE: about 38.9 ohms
- Mean spatial pattern correlation: about 0.91

This suggests that the prototype can recover the overall local resistance pattern from limited measurements, although exact single-node reconstruction is still imperfect.

## Generated Files

- Dataset:
  - `data/resistance/resistance_inverse_synthetic.npz`
- Model checkpoint:
  - `resistance_results/checkpoints/best_inverse_model.pt`
- Test metrics:
  - `resistance_results/test_metrics.json`
- Test predictions:
  - `resistance_results/test_sample_predictions.csv`
- Visualization:
  - `resistance_results/figures/resistance_inverse_examples.png`
- Low/high current sensitivity analysis:
  - `resistance_results/sensitivity/low_vs_high_current_sensitivity.png`

## Why Low Reference Is Easier Than High Reference

Under a fixed-voltage setup, current is proportional to conductance:

```text
I = V / R
```

The sensitivity of current to a resistance change is:

```text
dI/dR = -V / R^2
```

Therefore, if low reference is 1 ohm and high reference is 100 ohms, current measurements are about 10,000 times more sensitive near the low-resistance state than near the high-resistance state. This means a model may localize low-resistance regions well while high-resistance regions remain uncertain. That is a physical identifiability issue, not just a random-forest issue.

## What We Need From Simon / Esther Next

To move from a teaching simulator to a research-relevant prototype, we need:

- The real array layout: 2x2, 3x3, 4x4, or larger.
- Which signals can actually be measured experimentally:
  - row values
  - column values
  - path values
  - whole-array equivalent resistance
  - time-series measurements
- Whether Simon already has a circuit model that maps local resistance values to measurable total/equivalent resistance values.
- Esther's experimental data format:
  - CSV, JSON, MATLAB, Excel, or other
  - columns and units
  - repeated measurements over time
  - whether there is any ground-truth imaging or local annotation
- Whether biological labels exist:
  - cell coverage
  - growth stage
  - abnormal region
  - treatment/control condition

## Research Directions

Potential research questions:

- How many measurements are needed to reconstruct a useful local resistance map?
- Which measurement design is most informative: rows/columns, diagonals, random paths, or adaptive paths?
- Can a model trained on simulated circuit data transfer to real experimental measurements?
- Can uncertainty identify regions that need extra measurement?
- Can reconstructed resistance maps detect abnormal cell growth earlier than aggregate resistance alone?
