from __future__ import annotations

import unittest

import torch

from dynamic_impedance import CircuitGraph, DynamicImpedanceGNN, MeasurementBatch, pairwise_row_column_protocol
from dynamic_impedance.data_schema import PhysicalParameterBatch
from dynamic_impedance.mna import ac_pair_currents, dc_pair_currents
from dynamic_impedance.physics_losses import total_inverse_loss


class TestSchemaAndModel(unittest.TestCase):
    def test_dc_batch_disables_l_and_c_training_mask(self) -> None:
        graph = CircuitGraph.row_column_array(2)
        protocol = pairwise_row_column_protocol(2)
        resistance = torch.tensor([[1.0, 1.0, 1.0, 100.0]])
        current = dc_pair_currents(graph, resistance, protocol)
        batch = MeasurementBatch.from_dc_currents(current, protocol)
        model = DynamicImpedanceGNN(graph, hidden_size=16, message_steps=2, dropout=0)
        output = model(batch)
        self.assertEqual(output["resistance"].shape, resistance.shape)
        self.assertEqual(output["rlc_active_mask"].tolist(), [True, False, False])

    def test_dc_loss_backpropagates(self) -> None:
        graph = CircuitGraph.row_column_array(2)
        protocol = pairwise_row_column_protocol(2)
        resistance = torch.tensor([[1.0, 1.0, 1.0, 100.0]])
        current = dc_pair_currents(graph, resistance, protocol)
        batch = MeasurementBatch.from_dc_currents(current, protocol)
        model = DynamicImpedanceGNN(graph, hidden_size=16, message_steps=2, dropout=0)
        loss, _ = total_inverse_loss(model, graph, batch, PhysicalParameterBatch(resistance))
        loss.backward()
        self.assertIsNotNone(model.parameter_head[-1].weight.grad)
        self.assertTrue(torch.isfinite(model.parameter_head[-1].weight.grad).all())

    def test_ac_batch_activates_all_parameter_heads(self) -> None:
        graph = CircuitGraph.row_column_array(2)
        protocol = pairwise_row_column_protocol(2)
        resistance = torch.ones(1, 4)
        inductance = torch.full((1, 4), 1e-3)
        capacitance = torch.full((1, 4), 1e-6)
        frequency = torch.tensor([10.0, 100.0, 1000.0])
        current = ac_pair_currents(graph, resistance, inductance, capacitance, frequency, protocol)
        batch = MeasurementBatch.from_ac_currents(current, frequency, protocol)
        model = DynamicImpedanceGNN(graph, hidden_size=16, message_steps=1, dropout=0)
        output = model(batch)
        self.assertEqual(output["rlc_active_mask"].tolist(), [True, True, True])
        self.assertEqual(output["capacitance"].shape, resistance.shape)


if __name__ == "__main__":
    unittest.main()
