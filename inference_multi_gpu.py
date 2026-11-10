"""Independent prompt-parallel T2V inference, launched with torchrun."""

import argparse
import json
import os
from pathlib import Path


def select_generator_state(state, key=None):
    """Prefer generator by default, falling back to EMA-only checkpoints."""
    if key is None:
        key = next((candidate for candidate in ("generator", "generator_ema") if candidate in state), None)
    if key is None or key not in state:
        raise ValueError(f"Checkpoint has no {key or 'generator / generator_ema'} weights; available keys: {list(state)[:20]}")
    weights = state[key]
    if not isinstance(weights, dict) or "optimizer" in weights:
        raise ValueError("Expected complete generator weights; export training shards before inference")
    return key, {name.replace("_fsdp_wrapped_module.", ""): value for name, value in weights.items()}


def read_prompts(path):
    """A directory means all recursive .txt files; JSONL is opt-in by file."""
    path = Path(path)
    files = sorted(path.rglob("*.txt")) if path.is_dir() else [path]
    records = []
    used_names = set()
    for source in files:
        with source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                prompt = line.rstrip("\r\n")
                if not prompt.strip():
                    continue
                if source.suffix == ".jsonl":
                    row = json.loads(prompt)
                    prompt = row.get("caption", row.get("prompt"))
                    if not isinstance(prompt, str) or not prompt.strip():
                        raise ValueError(f"{source}:{line_number}: expected caption or prompt string")
                filename = f"{prompt}.mp4"
                if "/" in filename or "\x00" in filename:
                    raise ValueError(f"{source}:{line_number}: prompt contains '/' or NUL; cannot use it verbatim as a filename")
                # Identical names share one task so separate ranks cannot overwrite each other.
                if filename in used_names:
                    continue
                used_names.add(filename)
                records.append(dict(source=str(source), line=line_number, prompt=prompt,
                                    filename=filename))
    if not records:
        raise ValueError(f"No nonempty prompts found in {path}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", default="configs/causal_forcing_dmd_framewise.yaml")
    parser.add_argument("--checkpoint_path", help="Exported generator .pt file (not rank shards)")
    parser.add_argument("--data_path", default="prompts", help="TXT/JSONL file or directory of TXT files")
    parser.add_argument("--output_folder", default="outputs/multi_gpu")
    parser.add_argument("--num_output_frames", type=int, default=12, help="Latent frames; 12 gives 45 video frames")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Write manifest and check partition without CUDA/models")
    args = parser.parse_args()
    if args.num_output_frames < 1 or args.fps < 1:
        parser.error("frames and fps must be positive")
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    records = read_prompts(args.data_path)
    output = Path(args.output_folder)
    output.mkdir(parents=True, exist_ok=True)
    name_max = os.pathconf(output, "PC_NAME_MAX")
    for record in records:
        if len(os.fsencode(record["filename"])) > name_max:
            raise ValueError(f"{record['source']}:{record['line']}: complete prompt filename exceeds filesystem limit ({name_max} bytes); name was not truncated")
    if rank == 0:
        with (output / "manifest.jsonl").open("w", encoding="utf-8") as stream:
            for index, record in enumerate(records):
                stream.write(json.dumps(dict(record, index=index, seed=args.seed + index), ensure_ascii=False) + "\n")
    assigned = list(range(rank, len(records), world))
    pending = [i for i in assigned if args.overwrite or not (output / records[i]["filename"]).is_file()]
    print(f"[rank {rank}/{world}] total={len(records)}, assigned={len(assigned)}, pending={len(pending)}", flush=True)
    if args.dry_run or not pending:
        return

    import torch
    # Must precede pipeline imports: demo_utils.memory captures current_device at import time.
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    from omegaconf import OmegaConf
    from torchvision.io import write_video
    from tqdm import tqdm
    from pipeline import CausalInferencePipeline, CausalDiffusionInferencePipeline
    from demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb
    from utils.misc import set_seed

    torch.set_grad_enabled(False)
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(args.config_path))
    # Exported weights may have LoRA merged; use their matching generated configuration.
    if args.checkpoint_path:
        companion = Path(args.checkpoint_path).with_suffix(".config.json")
        if companion.is_file():
            config = OmegaConf.merge(config, OmegaConf.load(companion))
    block = config.num_frame_per_block
    adjusted = args.num_output_frames - int(config.independent_first_frame)
    if adjusted % block:
        raise ValueError("num_output_frames is incompatible with num_frame_per_block")
    cls = CausalInferencePipeline if "denoising_step_list" in config else CausalDiffusionInferencePipeline
    pipeline = cls(config, device=device)

    def load_weights(path, key, strict):
        state = torch.load(path, map_location="cpu", weights_only=False)
        key, weights = select_generator_state(state, key)
        print(f"[rank {rank}] Loading {path}: {key}", flush=True)
        result = pipeline.generator.load_state_dict(weights, strict=strict)
        if not strict and (result.unexpected_keys or any("lora" not in name for name in result.missing_keys)):
            raise ValueError(f"Incompatible base checkpoint: {result}")

    if not args.checkpoint_path and config.get("generator_ckpt"):
        load_weights(config.generator_ckpt, None, False)
    if args.checkpoint_path:
        load_weights(args.checkpoint_path, "generator_ema" if args.use_ema else None, True)
    pipeline = pipeline.to(dtype=torch.bfloat16).eval()
    if get_cuda_free_memory_gb(device) < 40:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    else:
        pipeline.text_encoder.to(device)
    pipeline.generator.to(device)
    pipeline.vae.to(device)

    for index in tqdm(pending, desc=f"GPU {local_rank}", position=local_rank):
        record = records[index]
        set_seed(args.seed + index)
        noise = torch.randn([1, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16)
        video = pipeline.inference(noise=noise, text_prompts=[record["prompt"]])
        frames = (video[0].permute(0, 2, 3, 1).clamp(0, 1) * 255).to(dtype=torch.uint8, device="cpu")
        target = output / record["filename"]
        temporary = output / f".inference-rank-{rank}.partial.mp4"
        write_video(str(temporary), frames, fps=args.fps)
        os.replace(temporary, target)
        pipeline.vae.model.clear_cache()
        del video, frames, noise


if __name__ == "__main__":
    main()
