import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", "--no-save", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--export-on-complete", action="store_true",
                        help="Export the final factorized checkpoint to inference weights on CPU")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default=None, help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--tf", action="store_true")

    args = parser.parse_args()

    from omegaconf import OmegaConf
    import wandb
    import trainer as trainers
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize
    config.tf = args.tf 
    # get the filename of config_path
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir if args.logdir is not None else config.get("logdir", os.path.join("logs", config_name))
    if args.resume is not None:
        config.resume = args.resume
    if args.max_steps is not None:
        config.max_train_steps = args.max_steps
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb

    trainer_types = {
        "diffusion": "DiffusionTrainer", "ode": "ODETrainer",
        "score_distillation": "ScoreDistillationTrainer",
        "consistency_distillation": "ConsistencyDistillationTrainer",
    }
    if config.trainer not in trainer_types:
        raise ValueError(f"Unknown trainer: {config.trainer}")
    trainer = getattr(trainers, trainer_types[config.trainer])(config)
    trainer.train()

    wandb.finish()

    if args.export_on_complete:
        import json
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        if rank == 0:
            print(f"训练完毕：step={trainer.step}", flush=True)
            if config.no_save or not getattr(config, "factorized_enabled", False):
                print("跳过自动合并：需要启用 factorized 并保存 checkpoint。", flush=True)
                return
            checkpoint = Path(config.logdir).resolve() / f"checkpoint_{trainer.step:07d}"
            metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
            output = checkpoint.parent / f"generator_step{trainer.step}.pt"
            # The export launches its own CPU process group, independent of training.
            export_env = {key: value for key, value in os.environ.items()
                          if key not in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                                         "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
                                         "MASTER_ADDR", "MASTER_PORT"}
                          and not key.startswith("TORCHELASTIC_")}
            subprocess.run([
                sys.executable, "-m", "torch.distributed.run", "--standalone",
                f"--nproc_per_node={metadata['world_size']}",
                "-m", "utils.export_distillation", "--checkpoint", str(checkpoint),
                "--output", str(output),
            ], env=export_env, check=True)
            print(f"合并完毕：{output}", flush=True)


if __name__ == "__main__":
    main()
