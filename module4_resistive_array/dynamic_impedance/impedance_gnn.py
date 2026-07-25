from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .circuit_graph import CircuitGraph
from .data_schema import MeasurementBatch


def _bounded_log_parameter(raw: torch.Tensor, minimum: float, maximum: float) -> tuple[torch.Tensor, torch.Tensor]:
    log_min = math.log(minimum)
    log_max = math.log(maximum)
    log_value = log_min + (log_max - log_min) * torch.sigmoid(raw)
    return torch.exp(log_value), log_value


class FrequencyMeasurementEncoder(nn.Module):
    """Encode variable-length complex spectra for each drive measurement."""

    def __init__(self, hidden_size: int, reference_frequency_hz: float = 1.0) -> None:
        super().__init__()
        self.reference_frequency_hz = reference_frequency_hz
        self.network = nn.Sequential(
            nn.Linear(8, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, batch: MeasurementBatch) -> torch.Tensor:
        if batch.mode == "transient":
            raise NotImplementedError("transient batches require the future DAE/time encoder")
        amplitude = torch.sqrt(batch.current_real.square() + batch.current_imag.square() + 1e-20)
        voltage_amplitude = torch.sqrt(batch.voltage_real.square() + batch.voltage_imag.square() + 1e-20)
        unit_real = batch.current_real / amplitude
        unit_imag = batch.current_imag / amplitude
        frequency = torch.log1p(batch.frequency_hz / self.reference_frequency_hz)
        frequency = frequency.unsqueeze(-1).expand_as(amplitude)
        dc_flag = torch.full_like(amplitude, 1.0 if batch.mode == "dc" else 0.0)
        features = torch.stack(
            [
                torch.log(amplitude.clamp_min(1e-12)),
                unit_real,
                unit_imag,
                frequency,
                torch.log(voltage_amplitude.clamp_min(1e-12)),
                batch.voltage_real / voltage_amplitude,
                batch.voltage_imag / voltage_amplitude,
                dc_flag,
            ],
            dim=-1,
        )
        encoded = self.network(features)
        weight = batch.mask.unsqueeze(-1).to(encoded.dtype)
        return (encoded * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1)


class DynamicImpedanceGNN(nn.Module):
    """Circuit-edge GNN that is DC-compatible and AC-frequency-ready."""

    def __init__(
        self,
        graph: CircuitGraph,
        hidden_size: int = 96,
        message_steps: int = 4,
        dropout: float = 0.1,
        resistance_range: tuple[float, float] = (0.1, 1e4),
        inductance_range: tuple[float, float] = (1e-9, 10.0),
        capacitance_range: tuple[float, float] = (1e-15, 1.0),
    ) -> None:
        super().__init__()
        self.graph = graph
        self.hidden_size = hidden_size
        self.message_steps = message_steps
        self.resistance_range = resistance_range
        self.inductance_range = inductance_range
        self.capacitance_range = capacitance_range
        self.frequency_encoder = FrequencyMeasurementEncoder(hidden_size)
        self.node_embedding = nn.Parameter(torch.randn(graph.n_nodes, hidden_size) * 0.02)
        self.edge_type_embedding = nn.Parameter(torch.randn(graph.n_edges, hidden_size) * 0.01)
        self.self_layers = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(message_steps))
        self.neighbor_layers = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(message_steps))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(message_steps))
        self.dropout = nn.Dropout(dropout)
        self.parameter_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, 3))
        self.uncertainty_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.GELU(), nn.Linear(hidden_size // 2, 3))
        self.high_state_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.GELU(), nn.Linear(hidden_size // 2, 1))
        self.global_log_gain = nn.Parameter(torch.zeros(()))
        self.global_phase = nn.Parameter(torch.zeros(()))
        self.offset_real = nn.Parameter(torch.zeros(()))
        self.offset_imag = nn.Parameter(torch.zeros(()))
        self.register_buffer("edge_adjacency", graph.edge_adjacency())
        self.register_buffer("edge_index", graph.edge_index.clone())

    def forward(self, batch: MeasurementBatch) -> dict[str, torch.Tensor | str]:
        if batch.n_measurements != self.graph.n_edges:
            raise ValueError(
                "V1 requires one row-column drive measurement per component edge; "
                f"received {batch.n_measurements} measurements for {self.graph.n_edges} edges"
            )
        state = self.frequency_encoder(batch)
        source_position = self.node_embedding[self.edge_index[0]]
        target_position = self.node_embedding[self.edge_index[1]]
        state = state + source_position.unsqueeze(0) + target_position.unsqueeze(0) + self.edge_type_embedding
        adjacency = self.edge_adjacency.to(dtype=state.dtype)
        for self_layer, neighbor_layer, norm in zip(self.self_layers, self.neighbor_layers, self.norms):
            neighbor = torch.einsum("ij,bjh->bih", adjacency, state)
            update = self_layer(state) + neighbor_layer(neighbor)
            state = norm(state + self.dropout(F.gelu(update)))
        raw = self.parameter_head(state)
        resistance, log_resistance = _bounded_log_parameter(raw[..., 0], *self.resistance_range)
        inductance, log_inductance = _bounded_log_parameter(raw[..., 1], *self.inductance_range)
        capacitance, log_capacitance = _bounded_log_parameter(raw[..., 2], *self.capacitance_range)
        uncertainty = F.softplus(self.uncertainty_head(state)) + 0.03
        return {
            "resistance": resistance,
            "inductance": inductance,
            "capacitance": capacitance,
            "log_resistance": log_resistance,
            "log_inductance": log_inductance,
            "log_capacitance": log_capacitance,
            "log_parameter_uncertainty": uncertainty,
            "high_state_logit": self.high_state_head(state).squeeze(-1),
            "mode": batch.mode,
            "rlc_active_mask": torch.tensor(
                [True, batch.mode == "ac", batch.mode == "ac"],
                dtype=torch.bool,
                device=state.device,
            ),
        }

    def calibrate_current(self, current: torch.Tensor, reference_scale: torch.Tensor) -> torch.Tensor:
        gain = torch.exp(self.global_log_gain.clamp(-0.5, 0.5))
        phase = 0.25 * torch.tanh(self.global_phase)
        rotation = torch.complex(torch.cos(phase), torch.sin(phase))
        offset = torch.complex(self.offset_real, self.offset_imag) * 0.05 * reference_scale
        return current * gain * rotation + offset

    def calibration_penalty(self) -> torch.Tensor:
        return (
            self.global_log_gain.square()
            + self.global_phase.square()
            + self.offset_real.square()
            + self.offset_imag.square()
        )
