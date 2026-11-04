from collections.abc import Mapping
import torch

from utils.pab_conditions import CONDITION_INDICES, CONDITION_MASKS, CONDITION_NAMES, MASK_INDICES


def decompose_conditions(outputs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Apply the three-factor Möbius/inclusion-exclusion transform.

    ``outputs`` has condition dimension first and follows CONDITION_NAMES.
    """
    if outputs.ndim < 1 or outputs.shape[0] != 8:
        raise ValueError("outputs must have shape [8, ...] in empty/P/A/B/PA/PB/AB/PAB order")
    components = {}
    for name, mask in CONDITION_MASKS.items():
        value = torch.zeros_like(outputs[0])
        submask = mask
        while True:
            sign = -1 if (mask.bit_count() - submask.bit_count()) % 2 else 1
            value = value + sign * outputs[MASK_INDICES[submask]]
            if submask == 0:
                break
            submask = (submask - 1) & mask
        components[name] = value
    return components


def reconstruct_full_condition(components: Mapping[str, torch.Tensor]) -> torch.Tensor:
    missing = set(CONDITION_NAMES) - set(components)
    if missing:
        raise ValueError(f"missing components: {sorted(missing)}")
    return sum((components[name] for name in CONDITION_NAMES[1:]), components["empty"])


def decomposition_report(outputs: torch.Tensor) -> dict[str, object]:
    components = decompose_conditions(outputs)
    reconstructed = reconstruct_full_condition(components)
    residual = reconstructed - outputs[CONDITION_INDICES["PAB"]]
    return {
        "conditions": outputs,
        "components": components,
        "reconstructed_full": reconstructed,
        "reconstruction_error": residual,
        "reconstruction_max_abs": residual.abs().amax().item(),
        "component_norms": {name: value.float().norm().item() for name, value in components.items()},
    }
