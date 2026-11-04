"""Command-line caption parsing with positional input and output paths."""

import argparse
import json

from .parser import CaptionParser, DEFAULT_PARSER_VERSION, parse_jsonl


def main():
    cli = argparse.ArgumentParser(description="Parse captions with CPU workers.")
    cli.add_argument("input")
    cli.add_argument("output")
    cli.add_argument("--rules")
    cli.add_argument("--parser-version", default=DEFAULT_PARSER_VERSION)
    cli.add_argument("--input-format", choices=("auto", "jsonl", "text"), default="auto")
    cli.add_argument("--workers", type=int, default=1)
    cli.add_argument("--chunksize", type=int, default=128)
    args = cli.parse_args()
    if args.workers < 1 or args.chunksize < 1:
        cli.error("--workers and --chunksize must be positive integers")
    parser = (CaptionParser.from_json(args.rules, args.parser_version)
              if args.rules else CaptionParser(parser_version=args.parser_version))
    counts = parse_jsonl(args.input, args.output, parser, args.input_format,
                         args.workers, args.chunksize)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
