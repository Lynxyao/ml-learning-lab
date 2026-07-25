from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


MeasurementMode = Literal["dc", "ac", "transient"]


@dataclass
class MeasurementBatch:
    """Unified terminal-measurement representation.

    Frequency-domain tensors use [batch, frequency, measurement]. Transient
    support reserves the same container but will use a dedicated time axis in a
    later backend; it is rejected by the current frequency encoder.
    """

    current_real: torch.Tensor
    current_imag: torch.Tensor
    frequency_hz: torch.Tensor
    voltage_real: torch.Tensor
    voltage_imag: torch.Tensor
    drive_pairs: torch.Tensor
    mask: torch.Tensor
    mode: MeasurementMode

    def __post_init__(self) -> None:
        expected = self.current_real.shape
        if self.current_real.ndim != 3:
            raise ValueError("current tensors must have shape [batch, frequency, measurement]")
        for name in ("current_imag", "voltage_real", "voltage_imag", "mask"):
            if getattr(self, name).shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        if self.frequency_hz.shape != expected[:2]:
            raise ValueError(f"frequency_hz must have shape {expected[:2]}")
        if self.drive_pairs.shape != (expected[2], 2):
            raise ValueError(f"drive_pairs must have shape [{expected[2]}, 2]")
        if self.mode not in ("dc", "ac", "transient"):
            raise ValueError(f"unsupported measurement mode: {self.mode}")
        if self.mode == "dc" and torch.any(self.frequency_hz != 0):
            raise ValueError("DC batches must use zero frequency")
        if self.mode == "ac" and torch.any(self.frequency_hz <= 0):
            raise ValueError("AC batches require strictly positive frequencies")

    @classmethod
    def from_dc_currents(
        cls,
        currents: torch.Tensor,
        drive_pairs: torch.Tensor,
        voltage: float = 5.0,
    ) -> "MeasurementBatch":
        if currents.ndim != 2:
            raise ValueError("DC currents must have shape [batch, measurement]")
        real = currents.unsqueeze(1)
        zeros = torch.zeros_like(real)
        return cls(
            current_real=real,
            current_imag=zeros,
            frequency_hz=torch.zeros(currents.shape[0], 1, dtype=currents.dtype, device=currents.device),
            voltage_real=torch.full_like(real, float(voltage)),
            voltage_imag=zeros,
            drive_pairs=drive_pairs.to(device=currents.device, dtype=torch.long),
            mask=torch.ones_like(real, dtype=torch.bool),
            mode="dc",
        )

    @classmethod
    def from_ac_currents(
        cls,
        currents: torch.Tensor,
        frequency_hz: torch.Tensor,
        drive_pairs: torch.Tensor,
        voltage: complex | torch.Tensor = 5.0 + 0.0j,
    ) -> "MeasurementBatch":
        if not torch.is_complex(currents) or currents.ndim != 3:
            raise ValueError("AC currents must be complex with shape [batch, frequency, measurement]")
        batch_size, n_frequencies, n_measurements = currents.shape
        frequency = frequency_hz.to(device=currents.device, dtype=currents.real.dtype)
        if frequency.ndim == 1:
            frequency = frequency.unsqueeze(0).expand(batch_size, -1)
        if frequency.shape != (batch_size, n_frequencies):
            raise ValueError("frequency_hz must have shape [frequency] or [batch, frequency]")
        voltage_tensor = torch.as_tensor(voltage, dtype=currents.dtype, device=currents.device)
        if voltage_tensor.ndim == 0:
            voltage_tensor = voltage_tensor.expand_as(currents)
        elif voltage_tensor.shape == (batch_size, n_frequencies):
            voltage_tensor = voltage_tensor.unsqueeze(-1).expand_as(currents)
        elif voltage_tensor.shape != currents.shape:
            raise ValueError("AC voltage must be scalar, [batch, frequency], or match currents")
        return cls(
            current_real=currents.real,
            current_imag=currents.imag,
            frequency_hz=frequency,
            voltage_real=voltage_tensor.real,
            voltage_imag=voltage_tensor.imag,
            drive_pairs=drive_pairs.to(device=currents.device, dtype=torch.long),
            mask=torch.ones_like(currents.real, dtype=torch.bool),
            mode="ac",
        )

    @property
    def complex_current(self) -> torch.Tensor:
        return torch.complex(self.current_real, self.current_imag)

    @property
    def complex_voltage(self) -> torch.Tensor:
        return torch.complex(self.voltage_real, self.voltage_imag)

    @property
    def batch_size(self) -> int:
        return self.current_real.shape[0]

    @property
    def n_frequencies(self) -> int:
        return self.current_real.shape[1]

    @property
    def n_measurements(self) -> int:
        return self.current_real.shape[2]

    def to(self, device: torch.device | str) -> "MeasurementBatch":
        return MeasurementBatch(
            current_real=self.current_real.to(device),
            current_imag=self.current_imag.to(device),
            frequency_hz=self.frequency_hz.to(device),
            voltage_real=self.voltage_real.to(device),
            voltage_imag=self.voltage_imag.to(device),
            drive_pairs=self.drive_pairs.to(device),
            mask=self.mask.to(device),
            mode=self.mode,
        )


@dataclass
class PhysicalParameterBatch:
    resistance: torch.Tensor
    inductance: torch.Tensor | None = None
    capacitance: torch.Tensor | None = None
    resistance_mask: torch.Tensor | None = None
    inductance_mask: torch.Tensor | None = None
    capacitance_mask: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "PhysicalParameterBatch":
        def move(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.to(device)

        return PhysicalParameterBatch(
            resistance=self.resistance.to(device),
            inductance=move(self.inductance),
            capacitance=move(self.capacitance),
            resistance_mask=move(self.resistance_mask),
            inductance_mask=move(self.inductance_mask),
            capacitance_mask=move(self.capacitance_mask),
        )
