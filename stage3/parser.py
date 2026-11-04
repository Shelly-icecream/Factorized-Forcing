"""Conservative, offline English caption parsing for Stage 3 experiments."""

from dataclasses import asdict, dataclass
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_PARSER_VERSION = "rules_en_v3_multi_action"
AMBIGUOUS_ACTION_NOUNS = {"row"}
NEGATION_RE = re.compile(r"\b(?:not|no|never|without|isn't|aren't|don't|doesn't|can't)\b", re.I)
ARTICLE_RE = re.compile(r"\b(?:a|an|the)\s+", re.I)
PREPOSITION_RE = re.compile(r"\b(?:in|on|at|through|near|by|under|over|inside|outside|along|across)\b", re.I)


@dataclass(frozen=True)
class Span:
    field: str
    start: int
    end: int
    text: str
    rule_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParseResult:
    caption: str
    person: str
    action: str
    background: str
    remaining: str
    spans: tuple[Span, ...]
    parse_status: str
    failure_reason: str | None
    parser_version: str
    conditions: dict[str, str]
    leakage_warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("remaining", None)
        result["spans"] = [span.to_dict() for span in self.spans]
        return result


def _flatten_rules(rules: dict[str, Any]) -> dict[str, tuple[tuple[str, str], ...]]:
    flattened = {}
    for field in ("person", "action", "background"):
        values = []
        for category, entries in rules.get(field, {}).items():
            values.extend((entry, f"{field}_{category}") for entry in entries)
            if field in {"person", "background"} and category != "weather":
                for entry in entries:
                    if entry.endswith(("s", "people", "children", "men", "women")):
                        continue
                    plural = (entry[:-1] + "ies" if entry.endswith("y") and entry[-2:] not in {"ay", "ey", "oy"}
                              else entry + "es" if entry.endswith(("ch", "sh", "x", "z")) else entry + "s")
                    values.append((plural, f"{field}_{category}_plural"))
        flattened[field] = tuple(sorted(values, key=lambda item: len(item[0]), reverse=True))
    return flattened


class CaptionParser:
    def __init__(self, rules: dict[str, Any] | None = None, parser_version=DEFAULT_PARSER_VERSION):
        self.raw_rules = rules or self._default_rules()
        self.rules = _flatten_rules(self.raw_rules)
        self.parser_version = parser_version
        self.known_nonhuman = tuple(self.raw_rules.get("person", {}).get("nonhuman_subject", ()))
        forms = self.raw_rules.get("verb_forms", {})
        self.verb_forms = {base: tuple(values) for base, values in forms.items()}
        self._compiled_rules = {}
        self._action_entries = self._build_action_entries()

    @classmethod
    def from_json(cls, path: str, parser_version=DEFAULT_PARSER_VERSION):
        with open(path, encoding="utf-8") as handle:
            return cls(json.load(handle), parser_version)

    @staticmethod
    def _default_rules():
        with (Path(__file__).with_name("rules_en_v2.json")).open(encoding="utf-8") as handle:
            return json.load(handle)

    def parse(self, caption: str) -> ParseResult:
        lowered = caption.casefold()
        spans: list[Span] = []
        failure = None
        if NEGATION_RE.search(lowered):
            failure = "negation_not_supported"

        actions = self._action_matches(lowered)
        backgrounds = self._background_matches(lowered, actions)
        subjects = self._subject_matches(lowered, actions, backgrounds)
        if len(subjects) != 1:
            failure = failure or ("person_missing" if not subjects else "multiple_subjects")
        if not actions:
            failure = failure or "action_missing"
        if len(backgrounds) != 1:
            failure = failure or ("background_missing" if not backgrounds else "multiple_backgrounds")

        if len(subjects) == 1 and actions and len(backgrounds) == 1:
            person_start, person_end, person_rule = subjects[0]
            action_start = min((start for start, _, _ in actions if start >= person_end),
                               default=actions[0][0])
            person_start, person_end = self._expand_person(caption, person_start, person_end, action_start, backgrounds)
            spans = [
                Span("person", person_start, person_end, caption[person_start:person_end].strip(), person_rule),
                *[Span("action", start, end, caption[start:end].strip(), rule)
                  for start, end, rule in actions],
                Span("background", backgrounds[0][0], backgrounds[0][1],
                     caption[backgrounds[0][0]:backgrounds[0][1]].strip(), backgrounds[0][2]),
            ]
            if self._overlaps(spans):
                failure = "overlapping_spans"

        if failure:
            diagnostic_spans = self._diagnostic_spans(caption, subjects, actions, backgrounds)
            return self._failed(caption, failure, tuple(diagnostic_spans))

        spans.sort(key=lambda item: item.start)
        remaining = self._complement(caption, spans)
        from .conditions import build_conditions
        conditions = build_conditions(caption, spans, remaining)
        leakage = self._leakage_warning(remaining)
        fields = {field: " ".join(span.text for span in spans if span.field == field)
                  for field in ("person", "action", "background")}
        return ParseResult(
            caption, fields["person"], fields["action"], fields["background"], remaining,
            tuple(spans), "ok", None, self.parser_version, conditions, leakage,
        )

    def _matches(self, text, entries):
        key = tuple(entries)
        patterns = self._compiled_rules.get(key)
        if patterns is None:
            patterns = []
            for phrase, rule_id in key:
                escaped = re.escape(phrase.casefold())
                if rule_id.endswith("complete_phrases") and re.search(r"\b(?:a|an|the)\s+\w+", phrase):
                    escaped = re.sub(
                        r"\\ (a|an|the)\\ ([a-z]+)",
                        r"\\ \1\\ (?:[a-z-]+\\ ){0,3}\2",
                        escaped,
                    )
                patterns.append((re.compile(rf"(?<!\w){escaped}(?!\w)"), rule_id))
            self._compiled_rules[key] = patterns
        candidates = []
        for pattern, rule_id in patterns:
            candidates.extend(
                (match.start(), match.end(), rule_id)
                for match in pattern.finditer(text)
                if not text[match.end():].startswith(("'s", "s'"))
            )
        selected = []
        for candidate in sorted(candidates, key=lambda x: (x[0], -(x[1] - x[0]))):
            if not any(self._overlap(candidate, old) for old in selected):
                selected.append(candidate)
        return sorted(selected)

    def _diagnostic_spans(self, caption, people, actions, backgrounds):
        spans = []
        if len(people) == 1:
            start, end, rule = people[0]
            start, end = self._expand_person(
                caption, start, end,
                min((action[0] for action in actions if action[0] >= end), default=len(caption)),
                backgrounds
            )
            spans.append(Span("person", start, end, caption[start:end].strip(), rule))
        for start, end, rule in actions:
            spans.append(Span("action", start, end, caption[start:end].strip(), rule))
        for start, end, rule in backgrounds:
            spans.append(Span("background", start, end, caption[start:end].strip(), rule))
        return spans

    def _subject_matches(self, text, actions, backgrounds):
        candidates = self._matches(text, self.rules["person"])
        candidates = [
            candidate for candidate in candidates
            if not any(action[0] <= candidate[0] and candidate[1] <= action[1]
                       for action in actions)
            and not any(background[0] <= candidate[0] < background[1]
                        for background in backgrounds)
            and text[candidate[0]:candidate[1]] != "camera"
        ]
        candidates = self._merge_conjoined_subjects(candidates, text)
        first_sentence_end = min(
            (index for index in (
                text.find("."),
                text.find("!"),
                text.find("?"),
            ) if index >= 0),
            default=len(text),
        )
        first_sentence = [candidate for candidate in candidates if candidate[0] < first_sentence_end]
        if first_sentence and any(action[0] < first_sentence_end for action in actions):
            candidates = first_sentence
        if len(candidates) > 1:
            first = candidates[0]
            inclusion = re.match(r",\s*including\b", text[first[1]:], re.I)
            if inclusion:
                stop = text.find(",", first[1] + inclusion.end())
                if stop >= 0:
                    candidates = [first] + [item for item in candidates[1:] if item[0] > stop]
        candidates = self._prefer_subject_by_following_action(text, candidates, actions)
        first_sentence = [candidate for candidate in candidates if candidate[0] < first_sentence_end]
        return first_sentence or candidates

    @staticmethod
    def _merge_conjoined_subjects(subjects, text):
        if len(subjects) < 2:
            return subjects
        merged = []
        index = 0
        while index < len(subjects):
            start, end, rule = subjects[index]
            last_end = end
            next_index = index + 1
            while next_index < len(subjects):
                next_start, next_end, _ = subjects[next_index]
                between = text[last_end:next_start]
                if not re.fullmatch(
                    r"(?:\s*,\s*(?:and\s+)?|\s+and\s+)"
                    r"(?:(?:a|an|the)\s+)?",
                    between,
                    re.I,
                ):
                    break
                last_end = next_end
                next_index += 1
            if next_index > index + 1:
                merged.append((start, last_end, "person_conjoined"))
            else:
                merged.append((start, end, rule))
            index = next_index
        return merged

    @staticmethod
    def _prefer_subject_by_following_action(text, subjects, actions):
        if len(subjects) < 2:
            return subjects

        scores = []
        for subject in subjects:
            score = 0
            for action in actions:
                if action[0] < subject[1]:
                    continue
                _, subject_end = CaptionParser._expand_person(
                    text, subject[0], subject[1], action[0],
                )
                between = text[subject_end:action[0]]
                if re.search(r"[.!?;\n]|\band\b", between, re.I):
                    continue
                if len(between) > 60:
                    continue
                if re.fullmatch(
                    r"(?:\s+|,)*(?:is|are|was|were|be|been|being|can|could|"
                    r"may|might|will|would|shall|should|does|did|quickly|"
                    r"slowly|slightly|then|now|also|still|just|around|"
                    r"forward|backward|together|down|up)*\s*",
                    between,
                    re.I,
                ):
                    score += 1
            scores.append((score, subject))
        best = max(score for score, _ in scores)
        if best == 0 or sum(score == best for score, _ in scores) != 1:
            return subjects
        return [subject for score, subject in scores if score == best]

    def _build_action_entries(self):
        entries = list(self.rules["action"])
        for phrase, rule_id in self.rules["action"]:
            words = phrase.split()
            if len(words) == 1 and words[0] not in self.verb_forms:
                base = words[0]
                if base in AMBIGUOUS_ACTION_NOUNS:
                    continue
                third_person = base[:-1] + "ies" if base.endswith("y") else (
                    base + "es" if base.endswith(("ch", "sh", "s", "x", "z", "o")) else base + "s"
                )
                progressive = base[:-1] + "ying" if base.endswith("ie") else (
                    base[:-1] + "ing" if base.endswith("e") and not base.endswith("ee")
                    else base + "ing"
                )
                entries.extend((
                    (third_person, f"{rule_id}_third_person"),
                    (base + "ed", f"{rule_id}_past"),
                    (progressive, f"{rule_id}_progressive"),
                ))
        for base, forms in self.verb_forms.items():
            entries.extend((form, f"action_verb_{base}") for form in forms)
        return tuple(entries)

    def _action_matches(self, text):
        matches = self._matches(text, self._action_entries)
        expanded = []
        for start, end, rule in matches:
            prefix = re.search(r"(?:\b(?:is|are|was|were|can|will|slowly|quickly)\s+)+$", text[:start])
            if prefix:
                start = prefix.start()
            suffix = re.match(
                r"(?:\s+(?:a|an|the|his|her|their|my|with)\b[^,.;]*?"
                r"(?=\s+(?:in|on|at|through|near|by|under|over|inside|outside|along|across)\b|[,.;]|$))?",
                text[end:],
            )
            if suffix and suffix.group(0):
                end += len(suffix.group(0))
            expanded.append((start, end, rule))
        return self._merge_related_actions(expanded, text)

    def _background_matches(self, text, actions):
        matches = self._matches(text, self.rules["background"])
        result = []
        for start, end, rule in matches:
            if any(start >= action[0] and start < action[1] for action in actions):
                continue
            if text[start:end] == "background" and not re.search(
                r"\b(?:background|setting)\s+(?:is|shows|has)\b", text[max(0, start - 5):end + 12]
            ) and not re.search(r"\b(?:purple|blue|red|green|white|black|solid|vibrant)\s+$", text[:start]):
                continue
            had_preposition = False
            prefix_match = re.search(
                r"\b(?:in|on|at|through|near|by|under|over|inside|outside|along|across)"
                r"\s+(?:(?:a|an|the)\s+)?"
                r"(?:(?!(?:in|on|at|through|near|by|under|over|inside|outside|along|across|and|while)\b)"
                r"[a-z]+(?:-[a-z]+)*\s+){0,6}$",
                text[:start],
                re.I,
            )
            if prefix_match and not any(
                self._overlap((prefix_match.start(), end), action) for action in actions
            ):
                start = prefix_match.start()
                had_preposition = True
            scene_definition = re.search(
                r"\b(?:the\s+)?(?:background|setting)\s+(?:is|shows|has)\s+"
                r"(?:(?:a|an|the)\s+)?(?:[a-z-]+[ ,]+){0,6}$", text[:start], re.I,
            )
            if scene_definition:
                start = scene_definition.start()
                had_preposition = True
            prefix = text[max(0, start - 16):start]
            if not had_preposition and not PREPOSITION_RE.search(prefix) and not rule.endswith("_weather"):
                scene_prefix = re.match(
                    r"\s*(?:(?:a|an|the)\s+)?[a-z0-9 ,'-]+?\s+"
                    r"(?P<verb>surrounds|contains|features|reveals|shows|"
                    r"displays)\b",
                    text[start:],
                    re.I,
                )
                if scene_prefix:
                    scene_start = start
                    verb_end = scene_start + scene_prefix.start("verb")
                    sentence_start = max(
                        text.rfind(".", 0, start),
                        text.rfind("!", 0, start),
                        text.rfind("?", 0, start),
                        text.rfind(";", 0, start),
                        text.rfind("\n", 0, start),
                    ) + 1
                    if re.match(r"\s*(?:a|an|the)\b", text[sentence_start:start], re.I):
                        start = sentence_start
                    end = verb_end
                    if end <= start:
                        continue
                else:
                    sentence_start = max(
                        text.rfind(".", 0, start),
                        text.rfind("!", 0, start),
                        text.rfind("?", 0, start),
                        text.rfind(";", 0, start),
                        text.rfind("\n", 0, start),
                    ) + 1
                    sentence_end_match = re.search(r"[.!?;\n]", text[start:])
                    sentence_end = start + sentence_end_match.start() if sentence_end_match else len(text)
                    comma = text.find(",", start, sentence_end)
                    clause_end = comma if comma >= 0 else sentence_end
                    clause = text[sentence_start:clause_end]
                    if (
                        not re.search(r"[.!?;\n]", text[sentence_start:start])
                        and start - sentence_start < 80
                        and re.search(r"\b(?:scene|setting)\b", text[sentence_start:end + 6], re.I)
                    ):
                        start = sentence_start
                        # Keep the noun phrase; later descriptive clauses are not its span.
                    else:
                        continue
            if rule.endswith("_weather"):
                head = re.match(r"\s+(?:day|night|morning|afternoon|evening|weather|sky)\b", text[end:])
                if head:
                    end += head.end()
                elif not any(other[0] >= end and re.fullmatch(r"[\s\w-]{0,50}", text[end:other[0]])
                             and not other[2].endswith("_weather") for other in matches):
                    continue
            result.append((start, end, rule))
        merged = []
        for current in sorted(result, key=lambda item: (item[0], -item[1])):
            if merged and current[0] < merged[-1][1]:
                previous = merged[-1]
                merged[-1] = (previous[0], max(previous[1], current[1]), previous[2])
                continue
            if merged and current[0] - merged[-1][1] <= 3 and (
                merged[-1][2].endswith("_weather") or current[2].endswith("_weather")
            ):
                merged[-1] = (min(merged[-1][0], current[0]), max(merged[-1][1], current[1]), "background_weather_scene")
            else:
                merged.append(current)
        return merged

    @staticmethod
    def _merge_related_actions(matches, text):
        if len(matches) <= 1:
            return matches
        matches = sorted(matches, key=lambda item: (item[0], -item[1]))
        merged = []
        for current in matches:
            # Expansion can already include another matched verb. Keep its coverage once.
            if merged and current[0] < merged[-1][1]:
                previous = merged[-1]
                merged[-1] = (previous[0], max(previous[1], current[1]), previous[2])
                continue
            if merged and current[0] <= merged[-1][1] + 12 and re.search(r"\b(?:while|and|holding|carrying)\b", text[merged[-1][1]:current[0]]):
                merged[-1] = (merged[-1][0], current[1], "action_combined")
            else:
                merged.append(current)
        return merged

    @staticmethod
    def _expand_person(caption, start, end, action_start, backgrounds=()):
        prefix_start = max(
            caption.rfind(".", 0, start),
            caption.rfind("!", 0, start),
            caption.rfind("?", 0, start),
            caption.rfind(";", 0, start),
            caption.rfind("\n", 0, start),
        ) + 1
        prefix = caption[prefix_start:start]
        boundaries = list(re.finditer(
            r"\b(?:of|featuring|showcasing|depicting|including|in|on|at|near|by|with|through|where)\s+|[,;:]\s*",
            prefix, re.I,
        ))
        local_start = prefix_start + (boundaries[-1].end() if boundaries else 0)
        group = re.search(r"\b(?:(?:a|the)\s+)?(?:[a-z-]+\s+){0,3}(?:group|crowd|thousands|hundreds)\s+of\s+[^,.;:]*$", prefix, re.I)
        if group:
            local_start = prefix_start + group.start()
        earlier_articles = list(ARTICLE_RE.finditer(caption, prefix_start, start))
        if earlier_articles:
            article = earlier_articles[-1]
            adjectives = caption[article.end():start]
            if "," in adjectives and re.fullmatch(r"[a-zA-Z -]+(?:,\s*[a-zA-Z -]+)+", adjectives) and not re.search(
                r"\b(?:of|in|on|at|with|featuring|showcasing|is|are|was|were)\b", adjectives, re.I,
            ):
                local_start = article.start()
        articles = list(ARTICLE_RE.finditer(caption, local_start, start))
        if articles:
            start = articles[-1].start()
        elif start - local_start <= 100:
            # Bare noun phrases still carry identity and appearance modifiers.
            start = local_start
        while start < end and caption[start].isspace():
            start += 1
        modifier_end = min(
            (background[0] for background in backgrounds if end <= background[0] < action_start),
            default=action_start,
        )
        sentence_stop = re.search(r"[.!?;\n]", caption[end:modifier_end])
        if sentence_stop:
            modifier_end = end + sentence_stop.start()
        between = caption[end:modifier_end]
        appositive = re.match(
            r",\s*(?:full of|with|wearing|dressed in)\b[^,.;]+(?=,|$)", between, re.I,
        )
        modifier = appositive or re.match(
            r"\s+(?:in|with|wearing|dressed in)\b[^,.;]+", between, re.I,
        )
        if modifier:
            description = modifier.group(0)
            predicate = re.search(
                r"\s+(?:is|are|was|were|has|have|as|who|which|(?!wearing\b)[a-z]+ing)\b", description, re.I,
            )
            end += predicate.start() if predicate else len(description)
        while end > start and caption[end - 1].isspace():
            end -= 1
        return start, end

    def _leakage_warning(self, remaining):
        lowered = remaining.casefold()
        leaked = []
        for field in ("person", "action", "background"):
            if self._matches(lowered, self.rules[field]):
                leaked.append(field)
        return "remaining_contains:" + ",".join(leaked) if leaked else None

    @staticmethod
    def _overlap(left, right):
        return left[0] < right[1] and right[0] < left[1]

    def _overlaps(self, spans):
        return any(self._overlap((a.start, a.end), (b.start, b.end))
                   for index, a in enumerate(spans) for b in spans[index + 1:])

    @staticmethod
    def _complement(caption, spans):
        covered = [False] * len(caption)
        for span in spans:
            for index in range(span.start, span.end):
                covered[index] = True
        text = "".join(char if not covered[index] else " " for index, char in enumerate(caption))
        return re.sub(r"\s+", " ", text).strip(" ,;:")

    def _failed(self, caption, reason, spans):
        from .conditions import build_conditions
        fields = {}
        for field in ("person", "action", "background"):
            values = [span.text for span in spans if span.field == field]
            if values:
                fields[field] = " | ".join(values)
        return ParseResult(
            caption, fields.get("person", ""), fields.get("action", ""),
            fields.get("background", ""), re.sub(r"\s+", " ", caption).strip(), spans,
            "ambiguous" if reason.startswith(("multiple", "overlapping")) else "unsupported",
            reason, self.parser_version, build_conditions(caption, spans, caption, valid=False),
        )


def parse_jsonl(input_path, output_path, parser, input_format="auto", workers=1, chunksize=128):
    """Stream captions, optionally using CPU workers with bounded batches.

    Call from an ``if __name__ == "__main__"`` guard when workers > 1.
    Output order is identical to input order.
    """
    if workers < 1 or chunksize < 1:
        raise ValueError("workers and chunksize must be positive integers")
    counts = Counter()
    if input_format not in {"auto", "jsonl", "text"}:
        raise ValueError("input_format must be one of: auto, jsonl, text")
    output_parent = __import__("pathlib").Path(output_path).parent
    output_parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, encoding="utf-8") as source, open(output_path, "w", encoding="utf-8") as target:
        captions = _read_captions(source, input_format)
        if workers == 1:
            results = (parser.parse(caption).to_dict() for caption in captions)
            _write_results(results, target, counts)
        else:
            from concurrent.futures import ProcessPoolExecutor
            from itertools import islice
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_init_worker,
                initargs=(parser.raw_rules, parser.parser_version, getattr(parser, "model", None)),
            ) as pool:
                while True:
                    batch = list(islice(captions, workers * chunksize))
                    if not batch:
                        break
                    results = pool.map(_parse_worker, batch, chunksize=chunksize)
                    _write_results(results, target, counts)
    return dict(counts)


def _read_captions(source, input_format):
    for line in source:
        if not line.strip():
            continue
        if input_format == "text":
            yield line.rstrip("\r\n")
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            if input_format == "jsonl":
                raise
            yield line.rstrip("\r\n")
        else:
            yield item["caption"] if isinstance(item, dict) else str(item)


def _write_results(results, target, counts):
    for result in results:
        counts[result["parse_status"]] += 1
        target.write(json.dumps(result, ensure_ascii=False) + "\n")


def _init_worker(rules, parser_version, model=None):
    global _worker_parser
    if model is None:
        _worker_parser = CaptionParser(rules, parser_version)
    else:
        from .spacy_parser import SpacyCaptionParser
        _worker_parser = SpacyCaptionParser(rules, parser_version, model)


def _parse_worker(caption):
    return _worker_parser.parse(caption).to_dict()


if __name__ == "__main__":
    import argparse
    cli = argparse.ArgumentParser(description="Parse captions with CPU workers.")
    cli.add_argument("--input", required=True)
    cli.add_argument("--output", required=True)
    cli.add_argument("--rules")
    cli.add_argument("--input-format", choices=("auto", "jsonl", "text"), default="auto")
    cli.add_argument("--workers", type=int, default=1)
    cli.add_argument("--chunksize", type=int, default=128)
    args = cli.parse_args()
    parser = CaptionParser.from_json(args.rules) if args.rules else CaptionParser()
    print(json.dumps(parse_jsonl(args.input, args.output, parser,
                                 args.input_format, args.workers, args.chunksize)))
