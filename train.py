import argparse
import os


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", "--no-save", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
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


if __name__ == "__main__":
    main()
