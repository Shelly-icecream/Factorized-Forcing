from collections.abc import Sequence
import re


CONDITION_NAMES = ("empty", "P", "A", "B", "PA", "PB", "AB", "PAB")
# Bit masks describe factor membership; tensor positions follow CONDITION_NAMES.
CONDITION_MASKS = {name: sum({"P": 1, "A": 2, "B": 4}[letter] for letter in name)
                   if name != "empty" else 0 for name in CONDITION_NAMES}
CONDITION_INDICES = {name: index for index, name in enumerate(CONDITION_NAMES)}
MASK_INDICES = {CONDITION_MASKS[name]: index for name, index in CONDITION_INDICES.items()}


def _join(parts: Sequence[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(part.strip(" ,;:") for part in parts if part.strip())).strip()


def build_conditions(caption, spans, remaining, valid=True) -> dict[str, str]:
    if not valid:
        return {}
    fields = {
        span.field: span.text
        for span in spans
        if span.field in {"person", "action", "background"}
    }
    if set(fields) != {"person", "action", "background"}:
        return {}
    parts = {"P": fields["person"], "A": fields["action"], "B": fields["background"], "R": remaining}
    return {
        "empty": _join((parts["R"],)),
        "P": _join((parts["P"], parts["R"])),
        "A": _join((parts["A"], parts["R"])),
        "B": _join((parts["B"], parts["R"])),
        "PA": _join((parts["P"], parts["A"], parts["R"])),
        "PB": _join((parts["P"], parts["B"], parts["R"])),
        "AB": _join((parts["A"], parts["B"], parts["R"])),
        "PAB": caption,
    }
