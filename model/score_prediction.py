import torch


def linear_flow_velocity_to_score(
    velocity: torch.Tensor,
    noisy: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Convert linear-flow velocity to score: -(x_t + (1-t)v) / t."""
    if torch.any(t <= 0):
        raise ValueError("linear-flow score conversion is undefined at t=0")
    while t.ndim < noisy.ndim:
        t = t.unsqueeze(-1)
    return -(noisy + (1 - t) * velocity) / t


def linear_flow_difference_to_score(
    velocity_difference: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Convert a velocity difference without duplicating the common -x_t/t term."""
    if torch.any(t <= 0):
        raise ValueError("linear-flow score conversion is undefined at t=0")
    while t.ndim < velocity_difference.ndim:
        t = t.unsqueeze(-1)
    return -(1 - t) * velocity_difference / t


def linear_flow_fake_target(
    detached_rollout: torch.Tensor,
    noisy: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Velocity target for a detached clean rollout under x_t=(1-t)x_0+t eps."""
    if torch.any(t <= 0) or torch.any(t >= 1):
        raise ValueError("linear-flow velocity target expects 0 < t < 1")
    while t.ndim < detached_rollout.ndim:
        t = t.unsqueeze(-1)
    return (noisy - detached_rollout) / t
