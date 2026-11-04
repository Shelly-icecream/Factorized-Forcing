"""Stage 3 parsing, score decomposition and condition-aligned training."""

from .conditions import CONDITION_NAMES, build_conditions
from .parser import CaptionParser, ParseResult

__all__ = ["CONDITION_NAMES", "CaptionParser", "ParseResult", "build_conditions",
           "decompose_conditions", "reconstruct_full_condition"]


def __getattr__(name):
    if name in {"decompose_conditions", "reconstruct_full_condition"}:
        from .decomposition import decompose_conditions, reconstruct_full_condition
        return {"decompose_conditions": decompose_conditions,
                "reconstruct_full_condition": reconstruct_full_condition}[name]
    raise AttributeError(name)
