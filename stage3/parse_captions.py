"""Command-line caption parsing with positional input and output paths."""

import argparse
import json

from .parser import parse_jsonl
from .spacy_parser import make_parser, DEFAULT_SPACY_MODEL


def main():
    cli = argparse.ArgumentParser(description="Parse captions with CPU workers.")
    cli.add_argument("input")
    cli.add_argument("output")
    cli.add_argument("--rules")
    cli.add_argument("--parser-version")
    cli.add_argument("--backend", choices=("spacy", "rules"), default="spacy")
    cli.add_argument("--spacy-model", default=DEFAULT_SPACY_MODEL)
    cli.add_argument("--input-format", choices=("auto", "jsonl", "text"), default="auto")
    cli.add_argument("--workers", type=int, default=1)
    cli.add_argument("--chunksize", type=int, default=128)
    args = cli.parse_args()
    if args.workers < 1 or args.chunksize < 1:
        cli.error("--workers and --chunksize must be positive integers")
    parser = make_parser(args.backend, args.rules, args.parser_version, args.spacy_model)
    counts = parse_jsonl(args.input, args.output, parser, args.input_format,
                         args.workers, args.chunksize)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
