"""Gather Stage 3 generator shards on CPU and export inference weights."""

import argparse
from copy import deepcopy
from datetime import timedelta
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor


def clean_name(name):
    return name.replace("_fsdp_wrapped_module.", "")


def merge_lora(state, alpha=None):
    """Merge this repository's LoRALinear: W + (alpha/rank) * up @ down."""
    result = dict(state)
    downs = {name[:-len(".lora_down.weight")]: value for name, value in state.items()
             if name.endswith(".lora_down.weight")}
    ups = {name[:-len(".lora_up.weight")]: value for name, value in state.items()
           if name.endswith(".lora_up.weight")}
    if downs.keys() != ups.keys():
        raise ValueError("incomplete LoRA down/up pairs in generator checkpoint")
    for prefix, down in downs.items():
        up = ups[prefix]
        weight = state.get(prefix + ".weight")
        if down.ndim != 2 or up.ndim != 2 or down.shape[0] < 1 or up.shape[1] != down.shape[0]:
            raise ValueError(f"invalid LoRA dimensions: {prefix}")
        if weight is None or weight.shape != (up.shape[0], down.shape[1]):
            raise ValueError(f"missing or incompatible base weight: {prefix}")
        rank = down.shape[0]
        # WanDiffusionWrapper uses `lora_alpha or lora_rank`.
        scale = float(alpha or rank) / rank
        if not math.isfinite(scale):
            raise ValueError("LoRA alpha must be finite")
        result[prefix + ".weight"] = (weight.float() + scale * (up.float() @ down.float())).to(weight.dtype)
        del result[prefix + ".lora_down.weight"]
        del result[prefix + ".lora_up.weight"]
    return result


def inference_config(signature, merged):
    config = deepcopy(signature)
    config.pop("dataset_sha256", None)
    config["generator_ckpt"] = None
    config["resume"] = None
    if merged:
        config.setdefault("model_kwargs", {})["lora_rank"] = 0
    return config


def _gather_flat(local, sizes):
    """Gather uneven CPU tensors to rank zero without serializing tensor objects."""
    rank, world = dist.get_rank(), dist.get_world_size()
    maximum = max(sizes)
    if maximum == 0:
        return [torch.empty(0, dtype=local.dtype) for _ in range(world)] if rank == 0 else None
    padded = torch.zeros(maximum, dtype=local.dtype, device="cpu")
    padded[:local.numel()].copy_(local.reshape(-1).cpu())
    received = [torch.empty_like(padded) for _ in range(world)] if rank == 0 else None
    dist.gather(padded, gather_list=received, dst=0)
    return received


def _gather_tensor(value):
    rank, world = dist.get_rank(), dist.get_world_size()
    if not isinstance(value, ShardedTensor):
        if not isinstance(value, torch.Tensor) or hasattr(value, "to_local"):
            raise ValueError("export supports tensor/ShardedTensor checkpoints, not DTensor")
        return value.detach().cpu() if rank == 0 else None
    metadata = value.metadata()
    sizes, placements = [0] * world, []
    for shard in metadata.shards_metadata:
        owner = shard.placement.rank()
        count = math.prod(shard.shard_sizes)
        placements.append((shard, owner, sizes[owner], count))
        sizes[owner] += count
    local = torch.empty(sizes[rank], dtype=value.dtype, device="cpu")
    for shard in value.local_shards():
        entries = [entry for entry in placements if entry[0] == shard.metadata]
        if len(entries) != 1:
            raise ValueError("invalid shard placement metadata")
        _, owner, offset, count = entries[0]
        if owner != rank:
            raise ValueError("local shard owner does not match export rank")
        local[offset:offset + count].copy_(shard.tensor.detach().reshape(-1).cpu())
    received = _gather_flat(local, sizes)
    if rank != 0:
        return None
    full = torch.empty(tuple(metadata.size), dtype=value.dtype, device="cpu")
    for shard, owner, offset, count in placements:
        destination = full
        for dimension, (start, length) in enumerate(zip(shard.shard_offsets, shard.shard_sizes)):
            destination = destination.narrow(dimension, start, length)
        destination.copy_(received[owner][offset:offset + count].reshape(shard.shard_sizes))
    return full


def _ema_overlay(generator, ema, full, signature):
    """Original FSDP parameter slices are flat, ordered by process-group rank."""
    rank, world = dist.get_rank(), dist.get_world_size()
    local = {clean_name(name): value.detach().cpu() for name, value in ema.items()}
    descriptions = [None] * world
    dist.all_gather_object(descriptions, {name: (value.numel(), str(value.dtype)) for name, value in local.items()})
    names = set(local)
    if any(set(description) != names for description in descriptions):
        raise ValueError("EMA parameter names differ across ranks")
    shapes = {clean_name(name): tuple(value.shape) for name, value in generator.items()}
    for name in sorted(names):
        if name not in shapes:
            raise ValueError(f"EMA parameter missing from generator: {name}")
        sizes = [description[name][0] for description in descriptions]
        if len({description[name][1] for description in descriptions}) != 1:
            raise ValueError(f"EMA dtype differs across ranks: {name}")
        count = math.prod(shapes[name])
        replicated = signature.get("sharding_strategy") == "no_shard"
        if replicated:
            if any(size != count for size in sizes):
                raise ValueError(f"incomplete replicated EMA: {name}")
            if rank == 0:
                full[name] = local[name].reshape(shapes[name]).to(full[name].dtype)
        else:
            if sum(sizes) != count:
                raise ValueError(f"EMA shards do not reconstruct parameter: {name}")
            received = _gather_flat(local[name], sizes)
            if rank == 0:
                full[name] = torch.cat([piece[:size] for piece, size in zip(received, sizes)]).reshape(shapes[name]).to(full[name].dtype)


def export_checkpoint(checkpoint, output, use_ema=False, keep_lora=False, dtype=torch.bfloat16):
    """All ranks call this with the original world size and a CPU Gloo group."""
    if not dist.is_initialized() or dist.get_backend() != "gloo":
        raise ValueError("export requires an initialized Gloo process group")
    rank, world = dist.get_rank(), dist.get_world_size()
    checkpoint, output = Path(checkpoint), Path(output)
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if metadata["world_size"] != world:
        raise ValueError(f"export requires {metadata['world_size']} ranks, got {world}")
    if metadata["signature"].get("sharding_strategy", "full") not in {"full", "no_shard"}:
        raise ValueError("CPU export currently supports full/no_shard checkpoints only")
    config_path = output.with_suffix(".config.json")
    if output.exists() or config_path.exists():
        raise FileExistsError(f"export output already exists: {output} or {config_path}")
    for index in range(world):
        if not (checkpoint / f"rank_{index:05d}.pt").is_file():
            raise FileNotFoundError(f"missing checkpoint shard rank_{index:05d}.pt")
    if rank == 0:
        print(f"Loading {world} training shards on CPU; exporting generator only.", flush=True)
    state = torch.load(checkpoint / f"rank_{rank:05d}.pt", map_location="cpu", weights_only=False)
    if state["step"] != metadata["step"] or state["signature"] != metadata["signature"]:
        raise ValueError("rank checkpoint and metadata do not match")
    generator, ema = state["generator"]["model"], state.get("ema")
    del state
    descriptions = [None] * world
    dist.all_gather_object(descriptions, (sorted(generator), bool(ema)))
    if any(description != descriptions[0] for description in descriptions):
        raise ValueError("generator keys or EMA availability differ across ranks")
    if use_ema and not ema:
        raise ValueError("this checkpoint has no generator EMA; omit --use-ema")
    full = {} if rank == 0 else None
    if rank == 0:
        print(f"Gathering {len(generator)} generator tensors.", flush=True)
    for name in sorted(generator):
        tensor = _gather_tensor(generator[name])
        if rank == 0:
            cleaned = clean_name(name)
            if cleaned in full:
                raise ValueError(f"duplicate normalized parameter: {cleaned}")
            full[cleaned] = tensor
    if use_ema:
        _ema_overlay(generator, ema, full, metadata["signature"])
    del generator, ema
    if rank == 0:
        print("Merging adapters and writing inference weights/config.", flush=True)
        if not keep_lora:
            full = merge_lora(full, metadata["signature"].get("model_kwargs", {}).get("lora_alpha"))
        full = {name: value.to(dtype=dtype) if value.is_floating_point() else value for name, value in full.items()}
        payload = {"generator": full, "export_metadata": {
            "step": metadata["step"], "source": str(checkpoint),
            "weights": "ema" if use_ema else "generator", "lora_merged": not keep_lora}}
        if use_ema:
            payload["generator_ema"] = full
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, output)
        temporary = config_path.with_suffix(config_path.suffix + ".tmp")
        temporary.write_text(json.dumps(inference_config(metadata["signature"], not keep_lora), indent=2), encoding="utf-8")
        os.replace(temporary, config_path)
        print(json.dumps({"checkpoint": str(output), "config": str(config_path),
                          "step": metadata["step"], "weights": payload["export_metadata"]}), flush=True)
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description="Export Stage 3 training shards to inference weights on CPU")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--keep-lora", action="store_true")
    parser.add_argument("--dtype", choices=("bfloat16", "float32", "float16"), default="bfloat16")
    args = parser.parse_args()
    if "RANK" not in os.environ:
        parser.error("launch with torchrun --standalone --nproc_per_node=<checkpoint world size> -m stage3.export")
    dist.init_process_group("gloo", timeout=timedelta(minutes=60))
    try:
        export_checkpoint(args.checkpoint, args.output, args.use_ema, args.keep_lora, getattr(torch, args.dtype))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
