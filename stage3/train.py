"""Standalone Stage 3 entry; data checks do not import Wan or its dependencies."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Train Stage 3 or validate parsed data")
    parser.add_argument("--config_path", default="configs/stage3_framewise.yaml")
    parser.add_argument("--data-path")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--parser-version")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--logdir")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--fsdp-smoke-test", action="store_true")
    args = parser.parse_args()
    if args.fsdp_smoke_test:
        from .smoke import run_fsdp_smoke
        print(json.dumps(run_fsdp_smoke()))
        return
    if args.smoke_test:
        from .smoke import run_smoke
        print(json.dumps(run_smoke()))
        return
    if args.check_data and args.data_path:
        from .dataset import ParsedCaptionDataset
        data = ParsedCaptionDataset(args.data_path, args.parser_version)
        print(json.dumps({"valid": len(data), "counts": dict(data.counts),
                          "parser_versions": data.parser_versions, "sha256": data.fingerprint}))
        return
    from omegaconf import OmegaConf
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(args.config_path))
    for argument, key in ((args.data_path, "data_path"), (args.resume, "resume"),
                          (args.max_steps, "max_train_steps"), (args.logdir, "logdir"),
                          (args.parser_version, "stage3_parser_version")):
        if argument is not None:
            config[key] = argument
    config.no_save = args.no_save
    if args.check_data:
        from .dataset import ParsedCaptionDataset
        from .trainer import validate_config
        validate_config(config)
        data = ParsedCaptionDataset(config.data_path, config.stage3_parser_version)
        print(json.dumps({"valid": len(data), "counts": dict(data.counts)}))
        return
    from .trainer import Trainer
    Trainer(config).train()


if __name__ == "__main__":
    main()
