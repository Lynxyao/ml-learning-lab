# Dynamic Impedance GNN

This package is the frequency-ready successor to the simulator-constrained
Hybrid GNN. The existing Hybrid GNN remains unchanged as a reproducible
baseline.

## Implemented now

- A unified DC/AC measurement schema with complex current and voltage channels.
- A legacy adapter for the existing real-valued 64-current CSV files.
- A physical circuit graph with junctions as nodes and components as edges.
- Independent incidence-matrix DC and complex AC modified nodal analysis.
- A frequency encoder and circuit-edge GNN with R, L, C, state, calibration,
  and uncertainty heads.
- DC mode masking: only resistance is trained from the current dataset; L and C
  remain explicitly untrained.
- Boundary-current, uncertainty-aware resistance, classification, passivity,
  and calibration losses.
- DC and AC parameter Jacobians.
- Analytical tests for Ohm's law, parallel resistance, series RLC phasors, and
  row-column networks.

## Not experimentally validated

The AC path is structurally operational and analytically tested, but no real
frequency/phase data have been supplied. The placeholder L and C values used by
the Jacobian demonstration are not sensor estimates. Transient DAE support is a
reserved future backend and is not implemented yet.

## Current DC training

```powershell
python module4_resistive_array\train_dynamic_impedance_dc.py `
  --data_dir data\resistance_8x8_simon_v2 `
  --output_dir resistance_results\dynamic_impedance_dc `
  --epochs 40 --batch_size 128 --forward_batch_size 16 `
  --synthetic_sparse_samples 4000 --synthetic_max_high_cells 8

python module4_resistive_array\test_dynamic_impedance_dc.py `
  --data_dir data\resistance_8x8_simon_v2 `
  --checkpoint resistance_results\dynamic_impedance_dc\dynamic_impedance_dc_best.pt `
  --output_dir resistance_results\dynamic_impedance_dc
```

## Tests

```powershell
$env:PYTHONPATH="module4_resistive_array"
python -m unittest discover -s module4_resistive_array\dynamic_impedance\tests -v
```

## Jacobian analysis

Current DC data:

```powershell
python module4_resistive_array\analyze_dynamic_impedance_jacobian.py --mode dc
```

Synthetic AC readiness check:

```powershell
python module4_resistive_array\analyze_dynamic_impedance_jacobian.py `
  --mode ac --parameter resistance --frequencies 10,100,1000,10000
```

The AC command checks software behavior only. It must not be presented as an
experimental RLC result.

The DC trainer adds sparse 1-8 high-cell maps by default because the supplied
training maps are much denser than the 1-3 high-cell test maps. Exact binary
test maps are excluded from this curriculum. Set `--synthetic_sparse_samples 0`
only for a quick software smoke test.
