"""Rank-local sharded checkpoints; resumes require the same world size."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
    StateDictType, ShardedStateDictConfig, ShardedOptimStateDictConfig)


def _context(module):
    if isinstance(module, FSDP):
        return FSDP.state_dict_type(module, StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
            ShardedOptimStateDictConfig(offload_to_cpu=True))
    return nullcontext()


def _pack(module, optimizer):
    with _context(module):
        return {"model": module.state_dict(), "optimizer": (
            FSDP.optim_state_dict(module, optimizer) if isinstance(module, FSDP)
            else optimizer.state_dict())}


def _restore(module, optimizer, state):
    with _context(module):
        module.load_state_dict(state["model"])
        optim_state = (FSDP.optim_state_dict_to_load(module, optimizer, state["optimizer"])
                       if isinstance(module, FSDP) else state["optimizer"])
        optimizer.load_state_dict(optim_state)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def save_checkpoint(path, model, generator_optimizer, fake_optimizer, stream,
                    step, signature, ema=None):
    path = Path(path)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    if (path / "metadata.json").exists():
        raise FileExistsError(f"completed checkpoint already exists: {path}")
    path.mkdir(parents=True, exist_ok=True)
    state = {"generator": _pack(model.generator, generator_optimizer),
             "fake": _pack(model.fake_score, fake_optimizer),
             "stream": stream.state_dict(), "rng": rng_state(), "step": step,
             "ema": ema, "signature": signature,
             "z": model.z_normalization_state_dict() if hasattr(model, "z_normalization_state_dict") else None}
    temporary = path / f"rank_{rank:05d}.pt.tmp"
    torch.save(state, temporary)
    os.replace(temporary, path / f"rank_{rank:05d}.pt")
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        temporary = path / "metadata.json.tmp"
        temporary.write_text(json.dumps({"world_size": world, "step": step, "signature": signature}, indent=2), encoding="utf-8")
        os.replace(temporary, path / "metadata.json")
    if dist.is_initialized():
        dist.barrier()


def load_checkpoint(path, model, generator_optimizer, fake_optimizer, stream, signature):
    path = Path(path)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata["world_size"] != world or metadata["signature"] != signature:
        raise ValueError("checkpoint world size, dataset or training configuration does not match")
    state = torch.load(path / f"rank_{rank:05d}.pt", map_location="cpu", weights_only=False)
    if state["signature"] != signature or state["step"] != metadata["step"]:
        raise ValueError("checkpoint rank file is inconsistent with its metadata")
    _restore(model.generator, generator_optimizer, state["generator"])
    _restore(model.fake_score, fake_optimizer, state["fake"])
    if state["z"] is not None:
        model.load_z_normalization_state_dict(state["z"])
    stream.load_state_dict(state["stream"])
    restore_rng(state["rng"])
    return state["step"], state["ema"]
