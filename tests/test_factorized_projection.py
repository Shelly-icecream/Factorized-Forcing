import unittest
import torch
from model.factorized_objective import remove_parallel_component


class ProjectionTests(unittest.TestCase):
    def test_partial_and_full_projection(self):
        v, r = torch.tensor([3., 4.]), torch.tensor([2., 0.])
        self.assertIs(remove_parallel_component(v, r, 0), v)
        torch.testing.assert_close(remove_parallel_component(v, r, .5), torch.tensor([1.5, 4.]))
        residual = remove_parallel_component(v, r, 1)
        torch.testing.assert_close(residual, torch.tensor([0., 4.]))
        self.assertEqual(torch.dot(residual, r).item(), 0.)

    def test_mask_zero_reference_and_validation(self):
        v, r = torch.tensor([3., 4.]), torch.tensor([2., 100.])
        mask = torch.tensor([True, False])
        residual = remove_parallel_component(v, r, 1, mask)
        self.assertEqual((residual[mask] * r[mask]).sum().item(), 0.)
        torch.testing.assert_close(remove_parallel_component(v, torch.zeros(2), 1), v)
        for alpha in (-1, 2, float('nan')):
            with self.assertRaises(ValueError):
                remove_parallel_component(v, r, alpha)
