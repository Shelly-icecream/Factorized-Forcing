"""Parameter-free action projection for generator text conditioning only."""
from functools import lru_cache
import math
import torch


@lru_cache(maxsize=4)
def _parser(version):
    if version == 'spacy_en_v1_multi_action':
        from prompt_processing.spacy_parser import SpacyCaptionParser
        return SpacyCaptionParser()
    from prompt_processing.parser import CaptionParser, DEFAULT_PARSER_VERSION
    if version != DEFAULT_PARSER_VERSION:
        raise ValueError(f'Unsupported action parser version: {version}')
    return CaptionParser()


@lru_cache(maxsize=4096)
def extract_action(text, version):
    parsed = _parser(version).parse(text)
    return parsed.action if parsed.parse_status == 'ok' else ''


def enhance_action_projection(context, actions, scale):
    """Actions are lists of valid (non-padding) encoder token embeddings."""
    if not math.isfinite(float(scale)) or scale < 0:
        raise ValueError('generator_action_scale must be finite and nonnegative')
    if scale == 0:
        return context
    result = context.clone()
    for index, action in enumerate(actions):
        if action is None or action.numel() == 0:
            continue
        direction = action.float().mean(dim=0)
        norm_sq = direction.square().sum()
        if norm_sq <= 1e-12:
            continue
        value = context[index].float()
        projection = (value @ direction).unsqueeze(-1) * direction / norm_sq
        result[index] = (value + scale * projection).to(context.dtype)
    return result
