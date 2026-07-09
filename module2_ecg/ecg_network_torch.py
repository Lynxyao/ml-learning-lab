# -*- coding: utf-8 -*-
"""PyTorch models for ECG time-series classification."""

import torch
import torch.nn as nn


class ECG1DCNN(nn.Module):
    """Lightweight 1D CNN baseline for fixed-length ECG beat segments."""

    def __init__(self, in_channels=1, num_classes=5, base_channels=32, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class ECGLSTM(nn.Module):
    """LSTM baseline for sequential ECG segments."""

    def __init__(self, in_channels=1, num_classes=5, hidden_size=96, num_layers=1, dropout=0.25):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        _, (hidden, _) = self.lstm(x)
        last = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(last)


class ECGCNNLSTM(nn.Module):
    """CNN feature extractor followed by BiLSTM temporal classifier."""

    def __init__(self, in_channels=1, num_classes=5, base_channels=32, hidden_size=96, dropout=0.25):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=base_channels * 2,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.transpose(1, 2)
        _, (hidden, _) = self.lstm(x)
        last = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(last)


def build_ecg_model(model_name, in_channels=1, num_classes=5):
    name = model_name.lower()
    if name == "cnn":
        return ECG1DCNN(in_channels=in_channels, num_classes=num_classes)
    if name == "lstm":
        return ECGLSTM(in_channels=in_channels, num_classes=num_classes)
    if name in {"cnn_lstm", "cnnlstm", "cnn-lstm"}:
        return ECGCNNLSTM(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unknown ECG model: {model_name}")

