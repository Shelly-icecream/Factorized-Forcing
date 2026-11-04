"""Prompt encoding cache keys and token-length checks for preprocessing tools."""

from collections.abc import Callable
import hashlib
import json

from utils.pab_conditions import CONDITION_NAMES


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


