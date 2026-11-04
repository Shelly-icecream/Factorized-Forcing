"""Reproducible reservoir sample comparison of rules and local spaCy parsing."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

from .parser import CaptionParser, _read_captions
from .spacy_parser import SpacyCaptionParser


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("input")
    cli.add_argument("output")
    cli.add_argument("--sample-size", type=int, default=3000)
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--input-format", choices=("auto", "text", "jsonl"), default="text")
    args = cli.parse_args()
    if args.sample_size < 1:
        cli.error("--sample-size must be positive")
    rng = random.Random(args.seed)
    sample = []
    total = 0
    with open(args.input, encoding="utf-8") as source:
        for total, caption in enumerate(_read_captions(source, args.input_format), 1):
            if total <= args.sample_size:
                sample.append(caption)
            else:
                index = rng.randrange(total)
                if index < args.sample_size:
                    sample[index] = caption
    old, new = CaptionParser(), SpacyCaptionParser()
    counts = {"rules": Counter(), "spacy": Counter()}
    transitions = Counter()
    examples = {"gained": [], "lost": [], "retained": []}
    for index, caption in enumerate(sample, 1):
        before, after = old.parse(caption), new.parse(caption)
        for name, result in (("rules", before), ("spacy", after)):
            counts[name][result.failure_reason or "ok"] += 1
        was_ok, is_ok = before.parse_status == "ok", after.parse_status == "ok"
        transitions[f"{was_ok}->{is_ok}"] += 1
        group = "retained" if was_ok and is_ok else "gained" if is_ok else "lost" if was_ok else None
        if group and len(examples[group]) < 15:
            examples[group].append({"before": before.to_dict(), "after": after.to_dict()})
        if index % 500 == 0:
            print(f"Processed {index}/{len(sample)}", flush=True)
    report = {
        "input": args.input, "total_captions": total, "sample_size": len(sample),
        "seed": args.seed, "rules_version": old.parser_version,
        "spacy_version": new.parser_version, "model": new.model,
        "model_version": new.nlp.meta.get("version"),
        "spacy_library_version": __import__("spacy").__version__,
        "counts": {name: dict(value) for name, value in counts.items()},
        "transitions": dict(transitions), "examples": examples,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
