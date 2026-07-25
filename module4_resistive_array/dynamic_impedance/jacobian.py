from __future__ import annotations

from typing import Literal

import torch

from .circuit_graph import CircuitGraph
from .mna import ac_pair_currents, dc_pair_currents


ParameterName = Literal["resistance", "inductance", "capacitance"]


def measurement_parameter_jacobian(
    graph: CircuitGraph,
    resistance: torch.Tensor,
    drive_pairs: torch.Tensor,
    voltage: float = 5.0,
    frequency_hz: torch.Tensor | None = None,
    inductance: torch.Tensor | None = None,
    capacitance: torch.Tensor | None = None,
    parameter: ParameterName = "resistance",
) -> torch.Tensor:
    """Return d measurement / d log(parameter) for one physical map.

    DC output has [measurement, edge]. AC output separates real and imaginary
    channels and has [2, frequency, measurement, edge].
    """
    if resistance.ndim != 1 or resistance.shape[0] != graph.n_edges:
        raise ValueError("resistance must contain one edge map")
    if frequency_hz is None:
        if parameter != "resistance":
            raise ValueError("DC measurements only support resistance Jacobians")
        log_parameter = resistance.log().detach().requires_grad_(True)

        def dc_forward(log_value: torch.Tensor) -> torch.Tensor:
            return dc_pair_currents(graph, log_value.exp().unsqueeze(0), drive_pairs, voltage=voltage)[0]

        return torch.autograd.functional.jacobian(dc_forward, log_parameter, vectorize=True)

    if inductance is None or capacitance is None:
        raise ValueError("AC Jacobians require inductance and capacitance")
    base = {
        "resistance": resistance,
        "inductance": inductance,
        "capacitance": capacitance,
    }
    selected = base[parameter]
    if torch.any(selected <= 0):
        raise ValueError("log-parameter Jacobians require positive parameter values")
    log_parameter = selected.log().detach().requires_grad_(True)

    def ac_forward(log_value: torch.Tensor) -> torch.Tensor:
        values = {key: value for key, value in base.items()}
        values[parameter] = log_value.exp()
        current = ac_pair_currents(
            graph,
            values["resistance"].unsqueeze(0),
            values["inductance"].unsqueeze(0),
            values["capacitance"].unsqueeze(0),
            frequency_hz,
            drive_pairs,
            voltage=complex(voltage, 0),
        )[0]
        return torch.stack([current.real, current.imag], dim=0)

    return torch.autograd.functional.jacobian(ac_forward, log_parameter, vectorize=True)
