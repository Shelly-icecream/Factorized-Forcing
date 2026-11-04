import argparse
from collections import Counter, defaultdict
import json
import re

from prompt_processing.parser import CaptionParser
from .spacy_parser import make_parser, DEFAULT_SPACY_MODEL


def audit_jsonl(input_path: str, parser: CaptionParser, example_limit: int = 3) -> dict:
    totals = Counter()
    rule_hits = Counter()
    failures = Counter()
    unmatched = Counter()
    unmatched_examples = defaultdict(list)
    leakage = Counter()
    with open(input_path, encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            caption = item["caption"] if isinstance(item, dict) else str(item)
            result = parser.parse(caption)
            totals[result.parse_status] += 1
            if result.failure_reason:
                failures[result.failure_reason] += 1
            for span in result.spans:
                rule_hits[span.rule_id] += 1
            if result.leakage_warning:
                leakage[result.leakage_warning] += 1
            for fragment in _candidate_fragments(result.remaining):
                unmatched[fragment] += 1
                if len(unmatched_examples[fragment]) < example_limit:
                    unmatched_examples[fragment].append(caption)
    return {
        "totals": dict(totals),
        "failure_reasons": dict(failures),
        "rule_hits": dict(rule_hits),
        "leakage_warnings": dict(leakage),
        "unmatched_fragments": [
            {"text": text, "count": count, "examples": unmatched_examples[text]}
            for text, count in unmatched.most_common()
        ],
        "candidate_dictionary_additions": [
            {"text": text, "count": count}
            for text, count in unmatched.most_common()
        ],
    }


def _candidate_fragments(remaining: str) -> list[str]:
    return [
        fragment.casefold()
        for fragment in re.findall(r"[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*)?", remaining)
        if len(fragment) > 2
    ]


def main():
    argument_parser = argparse.ArgumentParser(description="Audit P/A/B caption parsing coverage")
    argument_parser.add_argument("input_jsonl")
    argument_parser.add_argument("--rules", default="prompt_processing/rules.json")
    argument_parser.add_argument("--parser-version")
    argument_parser.add_argument("--backend", choices=("spacy", "rules"), default="spacy")
    argument_parser.add_argument("--spacy-model", default=DEFAULT_SPACY_MODEL)
    argument_parser.add_argument("--example-limit", type=int, default=3)
    args = argument_parser.parse_args()
    parser = make_parser(args.backend, args.rules, args.parser_version, args.spacy_model)
    print(json.dumps(audit_jsonl(args.input_jsonl, parser, args.example_limit), indent=2))


if __name__ == "__main__":
    main()
