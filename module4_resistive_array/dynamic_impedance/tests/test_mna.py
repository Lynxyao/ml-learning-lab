from __future__ import annotations

import math
import unittest

import torch

from dynamic_impedance.circuit_graph import CircuitGraph, pairwise_row_column_protocol
from dynamic_impedance.jacobian import measurement_parameter_jacobian
from dynamic_impedance.mna import ac_pair_currents, dc_pair_currents


class TestModifiedNodalAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        self.single_edge = CircuitGraph(
            n_nodes=2,
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_labels=("component",),
        )
        self.single_pair = torch.tensor([[0, 1]], dtype=torch.long)

    def test_single_resistor_matches_ohms_law(self) -> None:
        current = dc_pair_currents(self.single_edge, torch.tensor([[10.0]]), self.single_pair, voltage=5.0)
        torch.testing.assert_close(current, torch.tensor([[0.5]]), rtol=1e-6, atol=1e-7)

    def test_parallel_resistors_add_conductance(self) -> None:
        graph = CircuitGraph(2, torch.tensor([[0, 0], [1, 1]], dtype=torch.long))
        current = dc_pair_currents(graph, torch.tensor([[10.0, 20.0]]), self.single_pair, voltage=5.0)
        torch.testing.assert_close(current, torch.tensor([[0.75]]), rtol=1e-6, atol=1e-7)

    def test_series_rlc_matches_analytic_phasor(self) -> None:
        resistance = torch.tensor([[10.0]])
        inductance = torch.tensor([[0.1]])
        capacitance = torch.tensor([[1e-3]])
        frequency = torch.tensor([100.0])
        current = ac_pair_currents(
            self.single_edge,
            resistance,
            inductance,
            capacitance,
            frequency,
            self.single_pair,
            voltage=5.0 + 0.0j,
        )[0, 0, 0]
        omega = 2 * math.pi * 100
        expected = 5 / complex(10, omega * 0.1 - 1 / (omega * 1e-3))
        torch.testing.assert_close(current, torch.tensor(expected, dtype=current.dtype), rtol=1e-5, atol=1e-6)

    def test_row_column_dc_solver_matches_existing_solver(self) -> None:
        from circuit_physics import ideal_pairwise_terminal_currents

        graph = CircuitGraph.row_column_array(3)
        protocol = pairwise_row_column_protocol(3)
        resistance = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
        current = dc_pair_currents(graph, resistance, protocol, voltage=5.0)
        reference = ideal_pairwise_terminal_currents(resistance, voltage=5.0)
        torch.testing.assert_close(current, reference, rtol=1e-5, atol=1e-6)

    def test_dc_log_resistance_jacobian_shape(self) -> None:
        graph = CircuitGraph.row_column_array(2)
        protocol = pairwise_row_column_protocol(2)
        jacobian = measurement_parameter_jacobian(
            graph,
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            protocol,
        )
        self.assertEqual(jacobian.shape, (4, 4))
        self.assertTrue(torch.isfinite(jacobian).all())


if __name__ == "__main__":
    unittest.main()
