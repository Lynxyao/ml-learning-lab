from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class HybridPhysicsGNN(nn.Module):
    """Physics-proxy correction GNN for row-column resistance arrays.

    The measured current matrix is first converted to an equivalent-resistance
    proxy. Message passing then follows the circuit topology: two cells are
    neighbors when they share a row terminal or a column terminal. The network
    predicts a correction to the proxy, not an unconstrained resistance map.

    Shared gain and offset parameters provide a small calibration layer for
    future experimental fine-tuning. They are regularized toward the ideal
    simulator values and must not be interpreted as validated hardware physics.
    """

    def __init__(
        self,
        grid_size: int,
        log_current_mean: torch.Tensor,
        log_current_std: torch.Tensor,
        log_proxy_mean: torch.Tensor,
        log_proxy_std: torch.Tensor,
        current_scale: float,
        hidden_size: int = 128,
        message_steps: int = 4,
        dropout: float = 0.1,
        voltage: float = 5.0,
        prediction_min: float = 1.0,
        prediction_max: float = 110.0,
        max_log_correction: float = 5.0,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.n_cells = grid_size * grid_size
        self.hidden_size = hidden_size
        self.message_steps = message_steps
        self.voltage = voltage
        self.prediction_min = prediction_min
        self.prediction_max = prediction_max
        self.max_log_correction = max_log_correction

        self.register_buffer("log_current_mean", log_current_mean.reshape(1, self.n_cells).float())
        self.register_buffer("log_current_std", log_current_std.reshape(1, self.n_cells).float().clamp_min(1e-6))
        self.register_buffer("log_proxy_mean", log_proxy_mean.reshape(1, self.n_cells).float())
        self.register_buffer("log_proxy_std", log_proxy_std.reshape(1, self.n_cells).float().clamp_min(1e-6))
        self.register_buffer("current_scale", torch.tensor(float(current_scale), dtype=torch.float32))
        self.register_buffer("adjacency", self._build_rook_adjacency(grid_size))

        self.node_encoder = nn.Sequential(
            nn.Linear(5, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.row_embedding = nn.Parameter(torch.randn(grid_size, hidden_size) * 0.02)
        self.column_embedding = nn.Parameter(torch.randn(grid_size, hidden_size) * 0.02)
        self.self_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(message_steps)])
        self.neighbor_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(message_steps)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(message_steps)])
        self.dropout = nn.Dropout(dropout)

        self.correction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.high_state_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Start from the physical proxy rather than a random correction.
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

        # Dataset-level nuisance parameters for later hardware calibration.
        self.global_log_gain = nn.Parameter(torch.zeros(()))
        self.row_log_gain = nn.Parameter(torch.zeros(grid_size))
        self.column_log_gain = nn.Parameter(torch.zeros(grid_size))
        self.offset_raw = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _build_rook_adjacency(grid_size: int) -> torch.Tensor:
        n_cells = grid_size * grid_size
        adjacency = torch.zeros(n_cells, n_cells, dtype=torch.float32)
        for row in range(grid_size):
            for col in range(grid_size):
                idx = row * grid_size + col
                for other_col in range(grid_size):
                    if other_col != col:
                        adjacency[idx, row * grid_size + other_col] = 1.0
                for other_row in range(grid_size):
                    if other_row != row:
                        adjacency[idx, other_row * grid_size + col] = 1.0
        return adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)

    def equivalent_resistance_proxy(self, current: torch.Tensor) -> torch.Tensor:
        current = current.clamp_min(1e-8)
        # For a uniform n x n complete bipartite resistor network,
        # R_cell = R_eq * n^2 / (2n - 1).
        uniform_network_factor = self.n_cells / (2 * self.grid_size - 1)
        proxy = self.voltage / current * uniform_network_factor
        return proxy.clamp(self.prediction_min, self.prediction_max)

    def _node_features(self, current: torch.Tensor, proxy: torch.Tensor) -> torch.Tensor:
        batch_size = current.shape[0]
        log_current = torch.log(current.clamp_min(1e-8))
        log_proxy = torch.log(proxy)
        current_z = (log_current - self.log_current_mean) / self.log_current_std
        proxy_z = (log_proxy - self.log_proxy_mean) / self.log_proxy_std

        current_grid = current_z.reshape(batch_size, self.grid_size, self.grid_size)
        row_context = current_grid.mean(dim=2, keepdim=True).expand(-1, -1, self.grid_size)
        column_context = current_grid.mean(dim=1, keepdim=True).expand(-1, self.grid_size, -1)
        global_context = current_grid.mean(dim=(1, 2), keepdim=True).expand_as(current_grid)
        return torch.stack(
            [
                proxy_z,
                current_z,
                row_context.reshape(batch_size, -1),
                column_context.reshape(batch_size, -1),
                global_context.reshape(batch_size, -1),
            ],
            dim=-1,
        )

    def forward(self, current: torch.Tensor) -> dict[str, torch.Tensor]:
        if current.ndim != 2 or current.shape[1] != self.n_cells:
            raise ValueError(f"current must have shape [batch, {self.n_cells}]")

        proxy = self.equivalent_resistance_proxy(current)
        node_state = self.node_encoder(self._node_features(current, proxy))
        position = (
            self.row_embedding[:, None, :] + self.column_embedding[None, :, :]
        ).reshape(self.n_cells, self.hidden_size)
        node_state = node_state + position.unsqueeze(0)

        for self_layer, neighbor_layer, norm in zip(self.self_layers, self.neighbor_layers, self.norms):
            neighbor_state = torch.einsum("ij,bjh->bih", self.adjacency, node_state)
            update = self_layer(node_state) + neighbor_layer(neighbor_state)
            node_state = norm(node_state + self.dropout(F.gelu(update)))

        delta_log_r = self.max_log_correction * torch.tanh(self.correction_head(node_state).squeeze(-1))
        proxy_log_r = torch.log(proxy)
        pred_log_r = (proxy_log_r + delta_log_r).clamp(
            math.log(self.prediction_min),
            math.log(self.prediction_max),
        )
        uncertainty = (F.softplus(self.uncertainty_head(node_state).squeeze(-1)) + 0.03).clamp(max=2.5)
        high_state_logit = self.high_state_head(node_state).squeeze(-1)
        return {
            "resistance": torch.exp(pred_log_r),
            "log_resistance": pred_log_r,
            "proxy_resistance": proxy,
            "delta_log_resistance": delta_log_r,
            "log_resistance_uncertainty": uncertainty,
            "high_state_logit": high_state_logit,
        }

    def calibrate_forward_current(self, ideal_current: torch.Tensor) -> torch.Tensor:
        gain_log = (
            self.global_log_gain
            + self.row_log_gain[:, None]
            + self.column_log_gain[None, :]
        ).clamp(-0.5, 0.5)
        gain = torch.exp(gain_log).reshape(1, self.n_cells)
        offset = 0.05 * torch.tanh(self.offset_raw) * self.current_scale
        return (ideal_current * gain + offset).clamp_min(1e-8)

    def calibration_penalty(self) -> torch.Tensor:
        return (
            self.global_log_gain.square()
            + self.row_log_gain.square().mean()
            + self.column_log_gain.square().mean()
            + self.offset_raw.square()
        )

    @torch.no_grad()
    def calibration_summary(self) -> dict[str, float | list[float]]:
        return {
            "global_gain": float(torch.exp(self.global_log_gain).cpu()),
            "row_gain": torch.exp(self.row_log_gain).cpu().tolist(),
            "column_gain": torch.exp(self.column_log_gain).cpu().tolist(),
            "offset": float((0.05 * torch.tanh(self.offset_raw) * self.current_scale).cpu()),
        }
