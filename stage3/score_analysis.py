"""Frozen-teacher/fake analysis adapter; model-specific rollout stays outside this module."""

from collections.abc import Callable
import hashlib
import json
import torch

from .conditions import CONDITION_NAMES
from .decomposition import decomposition_report


class ConditionEncodingCache:
    """In-memory cache keyed by model/tokenizer/parser versions and exact text."""

    def __init__(self):
        self._values: dict[str, dict] = {}

    def key(
        self,
        condition: str,
        model_version: str,
        tokenizer_version: str,
        parser_version: str,
    ) -> str:
        payload = json.dumps(
            [model_version, tokenizer_version, parser_version, condition],
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get_or_encode(
        self,
        condition: str,
        encode_one: Callable[[str], dict],
        model_version: str,
        tokenizer_version: str,
        parser_version: str,
    ) -> dict:
        cache_key = self.key(condition, model_version, tokenizer_version, parser_version)
        if cache_key not in self._values:
            self._values[cache_key] = encode_one(condition)
        return self._values[cache_key]


def validate_tokenization(
    conditions: dict[str, str],
    tokenize: Callable[[str], object],
    max_length: int,
) -> dict[str, dict[str, object]]:
    if max_length < 1:
        raise ValueError("max_length must be positive")
    report = {}
    for name in CONDITION_NAMES:
        tokenized = tokenize(conditions[name])
        token_count = len(tokenized)
        report[name] = {
            "token_count": token_count,
            "truncated": token_count > max_length,
        }
    return report


@torch.no_grad()
def evaluate_conditions(
    noisy: torch.Tensor,
    timestep: torch.Tensor,
    conditions: dict[str, str],
    encode: Callable[[list[str]], dict],
    predict: Callable[[torch.Tensor, torch.Tensor, dict, torch.Tensor | None], torch.Tensor],
    mask: torch.Tensor | None = None,
    microbatch: int = 8,
) -> torch.Tensor:
    """Evaluate one caption at one shared noisy sample, returning [8, ...].

    noisy has batch size one; timestep and mask have a leading singleton
    batch dimension (a scalar timestep is also accepted). Each microbatch
    repeats the same sample, time and mask for its condition texts.
    """
    if tuple(conditions) != CONDITION_NAMES:
        raise ValueError("conditions must follow empty/P/A/B/PA/PB/AB/PAB order")
    if microbatch < 1:
        raise ValueError("microbatch must be positive")
    if noisy.ndim < 2 or noisy.shape[0] != 1:
        raise ValueError("evaluate_conditions expects one noisy sample with batch size 1")
    if not all(isinstance(value, str) for value in conditions.values()):
        raise ValueError("each condition must be a single prompt string")
    if timestep.ndim == 0:
        timestep = timestep.reshape(1)
    if timestep.shape[0] != 1 or (mask is not None and (mask.ndim == 0 or mask.shape[0] != 1)):
        raise ValueError("timestep and mask must have batch size 1")
    outputs = []
    names = list(CONDITION_NAMES)
    for start in range(0, len(names), microbatch):
        batch_names = names[start:start + microbatch]
        encoded = encode([conditions[name] for name in batch_names])
        count = len(batch_names)
        repeat = lambda value: value.repeat((count,) + (1,) * (value.ndim - 1))
        batch_noisy, batch_time = repeat(noisy), repeat(timestep)
        batch_mask = repeat(mask) if mask is not None else None
        prediction = predict(batch_noisy, batch_time, encoded, batch_mask)
        if not isinstance(prediction, torch.Tensor) or prediction.shape != batch_noisy.shape:
            raise ValueError("predict must return a tensor matching the repeated noisy sample shape")
        outputs.append(prediction)
    result = torch.cat(outputs, dim=0)
    if result.shape[0] != 8:
        raise ValueError("predict must return one condition result per encoded prompt")
    return result


def make_wan_predictor(model, prediction_type="score", scheduler=None):
    """Adapt Wan's (flow, x0) output. Score uses the scheduler's actual sigma.

    Pass the scheduler explicitly for wrapped models. Optional masks must be
    broadcastable to the prediction and zero out excluded entries.
    """
    if prediction_type not in {"flow", "x0", "score"}:
        raise ValueError("prediction_type must be flow, x0 or score")
    if prediction_type == "score" and scheduler is None:
        scheduler = model.get_scheduler()

    def predict(noisy, timestep, encoded, mask=None):
        if noisy.ndim != 5:
            raise ValueError("Wan expects noisy samples shaped [batch, frames, channels, height, width]")
        if timestep.ndim == 1:
            timestep = timestep[:, None].expand(-1, noisy.shape[1])
        if timestep.shape != noisy.shape[:2]:
            raise ValueError("Wan timestep must have shape [batch, frames]")
        if getattr(model, "uniform_timestep", False) and not torch.equal(
            timestep, timestep[:, :1].expand_as(timestep)
        ):
            raise ValueError("this Wan model requires the same timestep for every frame")
        flow, x0 = model(noisy_image_or_video=noisy, timestep=timestep,
                         conditional_dict=encoded)[:2]
        if prediction_type == "score":
            from .prediction import linear_flow_velocity_to_score
            times = scheduler.timesteps.to(device=timestep.device, dtype=torch.float64)
            indices = (times[:, None] - timestep.double().reshape(1, -1)).abs().argmin(dim=0)
            sigma = scheduler.sigmas.to(device=noisy.device, dtype=torch.float32)[indices]
            sigma = sigma.reshape(timestep.shape)
            result = linear_flow_velocity_to_score(flow.float(), noisy.float(), sigma)
        else:
            result = flow if prediction_type == "flow" else x0
        if mask is not None:
            if torch.broadcast_shapes(result.shape, mask.shape) != result.shape:
                raise ValueError("mask must broadcast to the prediction shape")
            result = result * mask.to(device=result.device, dtype=result.dtype)
        return result

    return predict


def analyze_teacher_and_fake(
    noisy: torch.Tensor,
    timestep: torch.Tensor,
    conditions: dict[str, str],
    encode: Callable[[list[str]], dict],
    teacher_predict: Callable,
    fake_predict: Callable,
    mask: torch.Tensor | None = None,
    microbatch: int = 8,
) -> dict[str, dict]:
    teacher = evaluate_conditions(noisy, timestep, conditions, encode, teacher_predict, mask, microbatch)
    fake = evaluate_conditions(noisy, timestep, conditions, encode, fake_predict, mask, microbatch)
    return {"real": decomposition_report(teacher), "fake": decomposition_report(fake)}
