import unittest
import torch
from model.factorized_objective import score_direction_metrics


class DirectionMetricsTests(unittest.TestCase):
    def test_cancellation_and_reference_ratios(self):
        p = torch.tensor([3., 4.], requires_grad=True)
        logs = score_direction_metrics({'P': p, 'A': -p, 'B': torch.zeros(2)}, p, 2*p)
        self.assertEqual(logs['score/weighted_norm/P'].item(), 5.)
        self.assertEqual(logs['score/relative_norm/P_to_baseline_cfg'].item(), .5)
        self.assertAlmostEqual(logs['score/cosine/P_vs_A'].item(), -1.)
        self.assertEqual(logs['score/sum_to_norm_sum'].item(), 0.)
        self.assertEqual(logs['score/cosine/P_vs_B_valid'].item(), 0.)
        self.assertTrue(all(torch.isfinite(v) and not v.requires_grad for v in logs.values()))

    def test_mask_and_zero_reference(self):
        p = torch.tensor([3., 100.])
        logs = score_direction_metrics({'P': p}, torch.zeros(2), p, torch.tensor([True, False]))
        self.assertEqual(logs['score/weighted_norm/P'].item(), 3.)
        self.assertEqual(logs['score/relative_norm/P_to_full_conditional_valid'].item(), 0.)
        self.assertEqual(logs['score/sum_to_norm_sum'].item(), 1.)


if __name__ == '__main__':
    unittest.main()
