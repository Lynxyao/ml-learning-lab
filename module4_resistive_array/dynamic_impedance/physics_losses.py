from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .circuit_graph import CircuitGraph
from .data_schema import MeasurementBatch, PhysicalParameterBatch
from .adapters import select_batch
from .impedance_gnn import DynamicImpedanceGNN
from .mna import ac_pair_currents, dc_pair_currents


@dataclass(frozen=True)
class PhysicsLossWeights:
    resistance: float = 1.0
    classification: float = 0.25
    boundary_current: float = 0.05
    calibration: float = 0.1
    passivity: float = 0.01


def resistance_uncertainty_loss(
    output: dict[str, torch.Tensor | str],
    target: PhysicalParameterBatch,
    high_threshold: float = 50.0,
    high_weight: float = 4.0,
) -> torch.Tensor:
    true_r = target.resistance.clamp_min(1e-12)
    error = output["log_resistance"] - torch.log(true_r)
    sigma = output["log_parameter_uncertainty"][..., 0]
    weight = torch.where(true_r > high_threshold, high_weight, 1.0)
    if target.resistance_mask is not None:
        weight = weight * target.resistance_mask.to(weight.dtype)
    return (weight * (0.5 * (error / sigma).square() + torch.log(sigma))).sum() / weight.sum().clamp_min(1)


def high_state_loss(
    output: dict[str, torch.Tensor | str],
    target: PhysicalParameterBatch,
    threshold: float = 50.0,
) -> torch.Tensor:
    true_high = (target.resistance > threshold).to(torch.float32)
    high_count = true_high.sum()
    low_count = true_high.numel() - high_count
    positive_weight = (low_count / high_count.clamp_min(1)).detach()
    return F.binary_cross_entropy_with_logits(
        output["high_state_logit"],
        true_high,
        pos_weight=positive_weight,
    )


def reconstruct_boundary_current(
    model: DynamicImpedanceGNN,
    graph: CircuitGraph,
    batch: MeasurementBatch,
    output: dict[str, torch.Tensor | str],
) -> torch.Tensor:
    if batch.mode == "dc":
        ideal = dc_pair_currents(
            graph,
            output["resistance"],
            batch.drive_pairs,
            voltage=batch.voltage_real[:, 0],
        ).unsqueeze(1)
        ideal = torch.complex(ideal, torch.zeros_like(ideal))
    elif batch.mode == "ac":
        ideal = ac_pair_currents(
            graph,
            output["resistance"],
            output["inductance"],
            output["capacitance"],
            batch.frequency_hz,
            batch.drive_pairs,
            voltage=batch.complex_voltage,
        )
    else:
        raise NotImplementedError("transient boundary reconstruction requires the future DAE backend")
    reference_scale = batch.complex_current.abs().mean().detach().clamp_min(1e-12)
    return model.calibrate_current(ideal, reference_scale)


def boundary_current_loss(predicted: torch.Tensor, batch: MeasurementBatch) -> torch.Tensor:
    measured = batch.complex_current
    mask = batch.mask.to(predicted.real.dtype)
    predicted_amplitude = predicted.abs().clamp_min(1e-12)
    measured_amplitude = measured.abs().clamp_min(1e-12)
    amplitude_error = (torch.log(predicted_amplitude) - torch.log(measured_amplitude)).square()
    predicted_unit = predicted / predicted_amplitude
    measured_unit = measured / measured_amplitude
    phase_error = (predicted_unit - measured_unit).abs().square()
    return ((amplitude_error + phase_error) * mask).sum() / mask.sum().clamp_min(1)


def passivity_loss(output: dict[str, torch.Tensor | str]) -> torch.Tensor:
    return (
        F.relu(-output["resistance"]).square().mean()
        + F.relu(-output["inductance"]).square().mean()
        + F.relu(-output["capacitance"]).square().mean()
    )


def kcl_residual_loss(
    graph: CircuitGraph,
    branch_current: torch.Tensor,
    external_injection: torch.Tensor,
    node_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """KCL residual for externally predicted branch currents and node injections."""
    incidence = graph.incidence_matrix(dtype=branch_current.dtype, device=branch_current.device)
    residual = torch.einsum("ne,...e->...n", incidence, branch_current) - external_injection
    if node_mask is not None:
        residual = residual * node_mask.to(residual.dtype)
        denominator = node_mask.sum().clamp_min(1)
    else:
        denominator = residual.numel()
    return residual.abs().square().sum() / denominator


def element_equation_loss(
    voltage_drop: torch.Tensor,
    branch_current: torch.Tensor,
    impedance: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    residual = voltage_drop - impedance * branch_current
    weight = torch.ones_like(residual.real) if mask is None else mask.to(residual.real.dtype)
    return (residual.abs().square() * weight).sum() / weight.sum().clamp_min(1)


def total_inverse_loss(
    model: DynamicImpedanceGNN,
    graph: CircuitGraph,
    batch: MeasurementBatch,
    target: PhysicalParameterBatch,
    weights: PhysicsLossWeights = PhysicsLossWeights(),
    high_threshold: float = 50.0,
    compute_boundary: bool = True,
    boundary_sample_count: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output = model(batch)
    resistance = resistance_uncertainty_loss(output, target, high_threshold=high_threshold)
    classification = high_state_loss(output, target, threshold=high_threshold)
    boundary = torch.zeros((), dtype=resistance.dtype, device=resistance.device)
    if compute_boundary and weights.boundary_current > 0:
        boundary_batch = batch
        boundary_output = output
        if boundary_sample_count is not None and boundary_sample_count < batch.batch_size:
            chosen = torch.randperm(batch.batch_size, device=batch.current_real.device)[:boundary_sample_count]
            boundary_batch = select_batch(batch, chosen)
            boundary_output = {
                key: value[chosen] if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch.batch_size else value
                for key, value in output.items()
            }
        boundary = boundary_current_loss(
            reconstruct_boundary_current(model, graph, boundary_batch, boundary_output),
            boundary_batch,
        )
    passive = passivity_loss(output)
    calibration = model.calibration_penalty()
    total = (
        weights.resistance * resistance
        + weights.classification * classification
        + weights.boundary_current * boundary
        + weights.passivity * passive
        + weights.calibration * calibration
    )
    return total, {
        "total": total,
        "resistance": resistance,
        "classification": classification,
        "boundary_current": boundary,
        "passivity": passive,
        "calibration": calibration,
    }
