"""Check baseline recovery against the x0-space DMD formula."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from model.factorized_objective import FactorizedObjective
from utils.pab_conditions import CONDITION_NAMES


class BaselineCFGTests(unittest.TestCase):
    def test_recovers_normalized_baseline_gradient(self):
        for real_scale, fake_scale in ((0., 0.), (3., 0.), (3., 1.)):
            with self.subTest(real_scale=real_scale, fake_scale=fake_scale):
                rollout = torch.tensor([1., 2.]).reshape(1, 2, 1, 1, 1).requires_grad_()
                noisy = torch.full_like(rollout, .3)
                sigma = torch.full((1, 2), .5)
                timestep = torch.full((1, 2), 500.)
                teacher = {name: float(i + 1) for i, name in enumerate(CONDITION_NAMES)}
                fake = {name: float(2 * i + 3) for i, name in enumerate(CONDITION_NAMES)}
                teacher['negative'], fake['negative'] = -2., -4.
                model = SimpleNamespace(
                    real_score=teacher, fake_score=fake, scheduler=None,
                    real_guidance_scale=real_scale, fake_guidance_scale=fake_scale,
                    args=SimpleNamespace(negative_prompt='negative'),
                    text_encoder=lambda text_prompts: {'texts': text_prompts},
                    _run_generator=lambda **kwargs: (rollout, None))
                objective = FactorizedObjective(model, list(rollout.shape), weights={
                    'empty': 1., 'PAB': 1., 'P': 0., 'A': 0., 'B': 0.})
                objective.noise_sample = lambda _: (noisy, None, timestep, sigma)

                def predictor(values, *args):
                    def predict(x, t, encoded, mask):
                        return torch.tensor([values[name] for name in encoded['texts']]).reshape(-1, 1, 1, 1, 1).expand_as(x)
                    return predict

                with patch('model.factorized_objective.make_wan_predictor', predictor):
                    loss, _ = objective.generator_loss(dict(zip(CONDITION_NAMES, CONDITION_NAMES)))
                loss.backward()
                # sigma=.5: x0 = 2*x_t + .5*score.
                real_x0 = .6 + .5 * (teacher['PAB'] + real_scale * (teacher['PAB'] - teacher['negative']))
                fake_x0 = .6 + .5 * (fake['PAB'] + fake_scale * (fake['PAB'] - fake['negative']))
                expected = (fake_x0 - real_x0) / (rollout.detach() - real_x0).abs().mean()
                torch.testing.assert_close(rollout.grad, expected.expand_as(rollout) / rollout.numel())


if __name__ == '__main__':
    unittest.main()
