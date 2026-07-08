from __future__ import annotations

import torch
from torch import nn


class FallRNN(nn.Module):
    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 48,
        num_layers: int = 1,
        dropout: float = 0.2,
        model: str = "gru",
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        rnn_cls = nn.GRU if model == "gru" else nn.LSTM
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * directions),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * directions, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last_step = out[:, -1, :]
        return self.classifier(last_step).squeeze(1)
