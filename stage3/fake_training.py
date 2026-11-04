"""Small, explicit helpers for condition-aligned fake-score updates."""

from dataclasses import dataclass
import torch

from .conditions import CONDITION_NAMES


@dataclass
class FakeSubsetBatch:
    subset: str
    condition: str
    rollout: torch.Tensor
    noisy: torch.Tensor
    noise: torch.Tensor
    timestep: torch.Tensor
    alpha: torch.Tensor
    sigma: torch.Tensor


def sample_subset(probabilities: dict[str, float], generator=None) -> str:
    if set(probabilities) != set(CONDITION_NAMES):
        raise ValueError("subset probabilities must contain all eight condition names")
    values = torch.tensor([probabilities[name] for name in CONDITION_NAMES], dtype=torch.float64)
    if not torch.isfinite(values).all() or torch.any(values < 0) or values.sum() <= 0:
        raise ValueError("subset probabilities must be finite, non-negative and have positive mass")
    values = values / values.sum()
    index = torch.multinomial(values, 1, generator=generator).item()
    return CONDITION_NAMES[index]


def make_fake_score_batch(
    subset: str,
    condition: str,
    rollout: torch.Tensor,
    noise: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    timestep: torch.Tensor,
    *,
    rollout_condition: str,
    rollout_subset: str,
) -> FakeSubsetBatch:
    """Keep subset labels separate from the exact text used for rollout.

    alpha/sigma may be scalars, per-example [B], or per-frame [B, F].
    Both the rollout and its condition metadata must come from the same sample.
    """
    if subset not in CONDITION_NAMES:
        raise ValueError("unknown condition subset")
    if subset != rollout_subset or condition != rollout_condition:
        raise ValueError("rollout and fake-score condition must be identical")
    if not isinstance(condition, str):
        raise ValueError("condition must be the prompt text")
    if rollout.shape != noise.shape:
        raise ValueError("rollout and noise must have the same shape")
    alpha = _coefficient(alpha, rollout)
    sigma = _coefficient(sigma, rollout)
    if torch.any(alpha < 0) or torch.any(sigma <= 0):
        raise ValueError("alpha must be non-negative and sigma must be positive")
    noisy = alpha * rollout.detach() + sigma * noise.detach()
    return FakeSubsetBatch(subset, condition, rollout.detach(), noisy,
                           noise.detach(), timestep, alpha, sigma)


def _coefficient(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype).detach()
    if not torch.isfinite(value).all():
        raise ValueError("noise coefficients must be finite")
    if value.ndim > reference.ndim or any(
        size not in (1, reference.shape[index]) for index, size in enumerate(value.shape)
    ):
        raise ValueError("noise coefficients must match leading batch/frame dimensions")
    return value.reshape(value.shape + (1,) * (reference.ndim - value.ndim))


def score_target_loss(prediction: torch.Tensor, batch: FakeSubsetBatch, sigma: torch.Tensor) -> torch.Tensor:
    """Denoising score matching target, -epsilon/sigma (not an epsilon prediction)."""
    if prediction.shape != batch.noise.shape:
        raise ValueError("prediction must match the noise shape")
    sigma = _coefficient(sigma, batch.noise)
    if torch.any(sigma <= 0):
        raise ValueError("epsilon score target is undefined at sigma=0")
    target = -batch.noise / sigma
    return torch.nn.functional.mse_loss(prediction, target)


def epsilon_target_loss(prediction: torch.Tensor, batch: FakeSubsetBatch, sigma: torch.Tensor) -> torch.Tensor:
    """Compatibility alias for score_target_loss; prediction must be a score."""
    return score_target_loss(prediction, batch, sigma)


def flow_target_loss(prediction: torch.Tensor, batch: FakeSubsetBatch) -> torch.Tensor:
    """Wan linear-flow objective: velocity = epsilon - detached clean rollout."""
    if prediction.shape != batch.rollout.shape:
        raise ValueError("prediction must match the rollout shape")
    if not torch.allclose(batch.alpha + batch.sigma, torch.ones_like(batch.alpha + batch.sigma)):
        raise ValueError("linear flow requires alpha = 1 - sigma")
    return torch.nn.functional.mse_loss(prediction, batch.noise - batch.rollout)
