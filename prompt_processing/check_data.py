"""Validate parsed P/A/B data without importing Wan, PyTorch or spaCy."""
import argparse
import json

from utils.pab_dataset import ParsedCaptionDataset


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("input")
    cli.add_argument("--parser-version")
    args = cli.parse_args()
    data = ParsedCaptionDataset(args.input, args.parser_version)
    print(json.dumps({"valid": len(data), "counts": dict(data.counts),
                      "parser_versions": data.parser_versions, "sha256": data.fingerprint}))


if __name__ == "__main__":
    main()
