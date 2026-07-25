from __future__ import annotations

import math

import torch

from .circuit_graph import CircuitGraph


def nodal_laplacian(graph: CircuitGraph, edge_admittance: torch.Tensor) -> torch.Tensor:
    """Assemble A diag(y) A^T for [batch, frequency, edge] admittance."""
    if edge_admittance.ndim != 3 or edge_admittance.shape[-1] != graph.n_edges:
        raise ValueError("edge_admittance must have shape [batch, frequency, edge]")
    incidence = graph.incidence_matrix(dtype=edge_admittance.dtype, device=edge_admittance.device)
    return torch.einsum("ne,bfe,me->bfnm", incidence, edge_admittance, incidence)


def _expand_voltage(
    voltage: float | complex | torch.Tensor,
    batch_size: int,
    n_frequencies: int,
    n_measurements: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    value = torch.as_tensor(voltage, dtype=dtype, device=device)
    if value.ndim == 0:
        return value.expand(batch_size, n_frequencies, n_measurements)
    if value.shape == (batch_size, n_frequencies, n_measurements):
        return value
    if n_frequencies == 1 and value.shape == (batch_size, n_measurements):
        return value.unsqueeze(1)
    if value.shape == (batch_size, n_frequencies):
        return value.unsqueeze(-1).expand(-1, -1, n_measurements)
    raise ValueError("voltage must be scalar, [batch, frequency], or [batch, frequency, measurement]")


def pair_currents_from_laplacian(
    laplacian: torch.Tensor,
    drive_pairs: torch.Tensor,
    voltage: float | complex | torch.Tensor,
) -> torch.Tensor:
    """Solve floating-node voltages and return source current for each drive pair."""
    if laplacian.ndim != 4 or laplacian.shape[-1] != laplacian.shape[-2]:
        raise ValueError("laplacian must have shape [batch, frequency, node, node]")
    batch_size, n_frequencies, n_nodes, _ = laplacian.shape
    pairs = drive_pairs.to(device=laplacian.device, dtype=torch.long)
    n_measurements = pairs.shape[0]
    drive_voltage = _expand_voltage(
        voltage,
        batch_size,
        n_frequencies,
        n_measurements,
        laplacian.dtype,
        laplacian.device,
    )
    all_nodes = torch.arange(n_nodes, device=laplacian.device)
    source_currents = []
    for measurement, pair in enumerate(pairs):
        driven = pair
        floating = all_nodes[(all_nodes != driven[0]) & (all_nodes != driven[1])]
        l_dd = laplacian.index_select(-2, driven).index_select(-1, driven)
        l_df = laplacian.index_select(-2, driven).index_select(-1, floating)
        l_fd = laplacian.index_select(-2, floating).index_select(-1, driven)
        l_ff = laplacian.index_select(-2, floating).index_select(-1, floating)
        source_v = torch.stack(
            [drive_voltage[:, :, measurement], torch.zeros_like(drive_voltage[:, :, measurement])],
            dim=-1,
        )
        floating_v = torch.linalg.solve(l_ff, -torch.matmul(l_fd, source_v.unsqueeze(-1)))
        driven_i = torch.matmul(l_dd, source_v.unsqueeze(-1)) + torch.matmul(l_df, floating_v)
        source_currents.append(driven_i[..., 0, 0])
    return torch.stack(source_currents, dim=-1)


def dc_pair_currents(
    graph: CircuitGraph,
    resistance: torch.Tensor,
    drive_pairs: torch.Tensor,
    voltage: float | torch.Tensor = 5.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return [batch, measurement] DC terminal currents."""
    if resistance.ndim != 2 or resistance.shape[1] != graph.n_edges:
        raise ValueError("resistance must have shape [batch, edge]")
    admittance = resistance.clamp_min(eps).reciprocal().unsqueeze(1)
    current = pair_currents_from_laplacian(nodal_laplacian(graph, admittance), drive_pairs, voltage)
    return current[:, 0]


def series_rlc_impedance(
    resistance: torch.Tensor,
    inductance: torch.Tensor,
    capacitance: torch.Tensor,
    frequency_hz: torch.Tensor,
    eps: float = 1e-20,
) -> torch.Tensor:
    """Series-RLC edge impedance; zero L/C means that element is absent."""
    if resistance.ndim != 2:
        raise ValueError("component tensors must have shape [batch, edge]")
    if inductance.shape != resistance.shape or capacitance.shape != resistance.shape:
        raise ValueError("R, L, and C tensors must have matching shapes")
    if frequency_hz.ndim == 1:
        frequency_hz = frequency_hz.unsqueeze(0).expand(resistance.shape[0], -1)
    if frequency_hz.ndim != 2 or frequency_hz.shape[0] != resistance.shape[0]:
        raise ValueError("frequency_hz must have shape [frequency] or [batch, frequency]")
    if torch.any(frequency_hz <= 0):
        raise ValueError("AC impedance requires positive frequencies")
    omega = 2 * math.pi * frequency_hz.unsqueeze(-1)
    real = resistance.unsqueeze(1).expand(-1, frequency_hz.shape[1], -1)
    imag = omega * inductance.unsqueeze(1)
    capacitive = torch.where(
        capacitance.unsqueeze(1) > 0,
        1.0 / (omega * capacitance.unsqueeze(1).clamp_min(eps)),
        torch.zeros_like(imag),
    )
    return torch.complex(real, imag - capacitive)


def ac_pair_currents(
    graph: CircuitGraph,
    resistance: torch.Tensor,
    inductance: torch.Tensor,
    capacitance: torch.Tensor,
    frequency_hz: torch.Tensor,
    drive_pairs: torch.Tensor,
    voltage: complex | torch.Tensor = 5.0 + 0.0j,
    eps: float = 1e-20,
) -> torch.Tensor:
    """Return complex [batch, frequency, measurement] terminal currents."""
    impedance = series_rlc_impedance(resistance, inductance, capacitance, frequency_hz, eps=eps)
    admittance = impedance.reciprocal()
    return pair_currents_from_laplacian(nodal_laplacian(graph, admittance), drive_pairs, voltage)
