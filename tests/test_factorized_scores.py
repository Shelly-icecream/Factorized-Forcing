import unittest
from types import SimpleNamespace

import torch

from utils.pab_conditions import CONDITION_NAMES, CONDITION_MASKS
from model.score_decomposition import decompose_conditions, reconstruct_full_condition
from model.condition_scores import evaluate_conditions, make_wan_predictor, analyze_teacher_and_fake
from model.fake_score_training import make_fake_score_batch, score_target_loss, flow_target_loss, sample_subset


class DecompositionTests(unittest.TestCase):
    def test_independent_factors_have_zero_interactions(self):
        outputs = torch.tensor([10., 11., 12., 14., 13., 15., 16., 17.])
        components = decompose_conditions(outputs)
        expected = [10., 1., 2., 4., 0., 0., 0., 0.]
        self.assertEqual([components[name].item() for name in CONDITION_NAMES], expected)

    def test_recovers_each_distinct_interaction(self):
        expected = {name: torch.tensor(float(i + 1)) for i, name in enumerate(CONDITION_NAMES)}
        outputs = torch.stack([
            sum(expected[sub] for sub in CONDITION_NAMES
                if CONDITION_MASKS[sub] & CONDITION_MASKS[name] == CONDITION_MASKS[sub])
            for name in CONDITION_NAMES
        ])
        actual = decompose_conditions(outputs)
        for name in CONDITION_NAMES:
            torch.testing.assert_close(actual[name], expected[name])
        torch.testing.assert_close(reconstruct_full_condition(actual), outputs[-1])


class EvaluationTests(unittest.TestCase):
    def test_microbatches_repeat_identical_latent_time_and_mask(self):
        noisy = torch.arange(6.).reshape(1, 2, 3).requires_grad_()
        time = torch.tensor([[0.2, 0.7]])
        mask = torch.tensor([[[1.], [0.]]])
        conditions = dict(zip(CONDITION_NAMES, map(str, range(8))))
        def encode(texts):
            return {"ids": torch.tensor([int(text) for text in texts])}
        for microbatch in (1, 3, 8, 12):
            with self.subTest(microbatch=microbatch):
                seen = []
                def predict(x, t, encoded, m):
                    count = len(encoded["ids"])
                    torch.testing.assert_close(x, noisy.detach().repeat(count, 1, 1))
                    torch.testing.assert_close(t, time.repeat(count, 1))
                    torch.testing.assert_close(m, mask.repeat(count, 1, 1))
                    self.assertFalse(torch.is_grad_enabled())
                    seen.extend(encoded["ids"].tolist())
                    return x + encoded["ids"][:, None, None]
                output = evaluate_conditions(noisy, time, conditions, encode, predict, mask, microbatch)
                self.assertEqual(seen, list(range(8)))
                torch.testing.assert_close(output, noisy.detach().repeat(8, 1, 1) + torch.arange(8.)[:, None, None])
                self.assertFalse(output.requires_grad)

    def test_invalid_prediction_shape_and_multi_sample_fail(self):
        conditions = dict(zip(CONDITION_NAMES, CONDITION_NAMES))
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            evaluate_conditions(torch.zeros(2, 3), torch.ones(2), conditions, lambda x: {}, lambda *x: x[0])
        with self.assertRaisesRegex(ValueError, "matching"):
            evaluate_conditions(torch.zeros(1, 3), torch.ones(1), conditions, lambda x: {}, lambda *x: torch.zeros(1, 3))

    def test_wan_score_uses_scheduler_sigma_and_frame_times(self):
        scheduler = SimpleNamespace(timesteps=torch.tensor([900., 250.]), sigmas=torch.tensor([0.8, 0.2]))
        calls = []
        def wan(**kwargs):
            x, t = kwargs["noisy_image_or_video"], kwargs["timestep"]
            calls.append(t.clone())
            return torch.full_like(x, 2.), torch.full_like(x, 3.)
        x = torch.ones(1, 2, 1, 1, 1)
        predict = make_wan_predictor(wan, "score", scheduler)
        result = predict(x, torch.tensor([[900., 250.]]), {})
        sigma = torch.tensor([0.8, 0.2]).reshape(1, 2, 1, 1, 1)
        torch.testing.assert_close(result, -(x + (1 - sigma) * 2) / sigma)
        torch.testing.assert_close(predict(x, torch.tensor([250.]), {}), torch.full_like(x, -13.))
        self.assertEqual(tuple(calls[-1].shape), (1, 2))
        torch.testing.assert_close(make_wan_predictor(wan, "flow")(x, torch.tensor([250.]), {}), torch.full_like(x, 2.))
        torch.testing.assert_close(make_wan_predictor(wan, "x0")(x, torch.tensor([250.]), {}), torch.full_like(x, 3.))

    def test_teacher_fake_analysis_has_no_artificial_interactions(self):
        conditions = dict(zip(CONDITION_NAMES, map(str, (0, 1, 2, 4, 3, 5, 6, 7))))
        encode = lambda texts: {"ids": torch.tensor([float(t) for t in texts])}
        predict = lambda x, t, e, m: x + e["ids"].reshape(-1, 1)
        report = analyze_teacher_and_fake(torch.zeros(1, 1), torch.tensor(0.5), conditions,
                                          encode, predict, predict, microbatch=3)
        for side in ("real", "fake"):
            for name in ("PA", "PB", "AB", "PAB"):
                self.assertEqual(report[side]["components"][name].item(), 0.)


class FakeTrainingTests(unittest.TestCase):
    def batch(self, **overrides):
        values = dict(subset="P", condition="A man", rollout=torch.ones(2, 3, 1, 1, 1, requires_grad=True),
                      noise=torch.full((2, 3, 1, 1, 1), 2.), alpha=torch.tensor([0.8, 0.6]),
                      sigma=torch.tensor([0.2, 0.4]), timestep=torch.tensor([200., 400.]),
                      rollout_condition="A man", rollout_subset="P")
        values.update(overrides)
        return make_fake_score_batch(**values)

    def test_labels_and_text_checked_separately(self):
        batch = self.batch()
        self.assertEqual(batch.subset, "P")
        self.assertEqual(batch.condition, "A man")
        self.assertFalse(batch.rollout.requires_grad)
        self.assertFalse(batch.noisy.requires_grad)
        torch.testing.assert_close(batch.noisy[:, 0, 0, 0, 0], torch.tensor([1.2, 1.4]))
        for overrides in ({"rollout_subset": "A"}, {"rollout_condition": "A woman"}, {"subset": "unknown"}):
            with self.assertRaises(ValueError):
                self.batch(**overrides)

    def test_framewise_schedule_and_score_target(self):
        sigma = torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
        batch = self.batch(alpha=1-sigma, sigma=sigma)
        target = -batch.noise / sigma[..., None, None, None]
        self.assertEqual(score_target_loss(target, batch, sigma).item(), 0.)
        self.assertEqual(flow_target_loss(batch.noise - batch.rollout, batch).item(), 0.)
        with self.assertRaises(ValueError):
            flow_target_loss(torch.zeros_like(batch.rollout), self.batch(alpha=torch.ones(2)))

    def test_one_fake_update_leaves_rollout_frozen(self):
        batch = self.batch()
        weight = torch.nn.Parameter(torch.tensor(0.))
        optimizer = torch.optim.SGD([weight], lr=0.1)
        optimizer.zero_grad()
        loss = flow_target_loss(weight.expand_as(batch.noise), batch)
        loss.backward()
        optimizer.step()
        self.assertGreater(weight.item(), 0.)
        self.assertIsNone(batch.rollout.grad)
        self.assertTrue(torch.isfinite(loss))

    def test_invalid_schedule_and_probabilities_fail(self):
        for overrides in ({"sigma": torch.zeros(2)}, {"sigma": torch.ones(3)},
                          {"alpha": torch.tensor(float("nan"))}):
            with self.assertRaises(ValueError):
                self.batch(**overrides)
        probabilities = dict.fromkeys(CONDITION_NAMES, 1.)
        probabilities["P"] = float("nan")
        with self.assertRaises(ValueError):
            sample_subset(probabilities)


if __name__ == "__main__":
    unittest.main()
