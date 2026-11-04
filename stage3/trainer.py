"""Dedicated distributed Stage 3 training loop."""

import json
from contextlib import nullcontext
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from .checkpoint import save_checkpoint, load_checkpoint
from .conditions import CONDITION_NAMES
from .dataset import ParsedCaptionDataset, CaptionStream
from .fake_training import sample_subset
from .training import Stage3Objective, validate_weights


def validate_config(config):
    if config.batch_size != 1 or list(config.image_or_video_shape)[0] != 1:
        raise ValueError("Stage 3 requires per-rank batch_size=1")
    if config.i2v:
        raise ValueError("Stage 3 currently supports text-to-video only")
    if not config.train_fake_score or not config.staged_fsdp_init:
        raise ValueError("Stage 3 requires train_fake_score=true and staged_fsdp_init=true")
    if config.denoising_loss_type != "flow":
        raise ValueError("Stage 3 fake training requires denoising_loss_type=flow")
    if config.stage3_microbatch < 1:
        raise ValueError("stage3_microbatch must be positive")
    if config.max_train_steps < 1 or config.dfake_gen_update_ratio < 1 or config.log_iters < 1:
        raise ValueError("step counts, update ratio and save interval must be positive")
    if not 0 <= config.ema_weight < 1:
        raise ValueError("ema_weight must be in [0, 1)")
    validate_weights(dict(config.stage3_component_weights))
    # Validate without consuming the training RNG.
    sample_subset(dict(config.stage3_subset_probabilities), torch.Generator().manual_seed(0))
    if any(config.stage3_subset_probabilities[name] <= 0 for name in CONDITION_NAMES):
        raise ValueError("all subsets need positive fake training probability for eight-condition analysis")


class Trainer:
    def __init__(self, config):
        validate_config(config)
        self.config = config
        self.dataset = ParsedCaptionDataset(config.data_path, config.stage3_parser_version)
        from utils.distributed import launch_distributed_job
        launch_distributed_job()
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.device = torch.device("cuda", torch.cuda.current_device())
        fingerprints = [None] * self.world
        dist.all_gather_object(fingerprints, self.dataset.fingerprint)
        if len(set(fingerprints)) != 1:
            raise ValueError("all ranks must read the same parsed-caption dataset")
        if config.resume:
            metadata = json.loads((Path(config.resume) / "metadata.json").read_text(encoding="utf-8"))
            if metadata["world_size"] != self.world:
                raise ValueError("resume requires the same world size as the checkpoint")
            config.seed = metadata["signature"]["seed"]
        elif config.seed == 0:
            seed = torch.tensor([random.SystemRandom().randrange(1, 2**31) if self.rank == 0 else 0], device=self.device)
            dist.broadcast(seed, src=0)
            config.seed = seed.item()
        random.seed(config.seed + self.rank)
        np.random.seed((config.seed + self.rank) % (2**32))
        torch.manual_seed(config.seed + self.rank)
        torch.cuda.manual_seed(config.seed + self.rank)
        self.stream = CaptionStream(self.dataset, self.rank, self.world, config.seed)
        # DMD supplies existing staged FSDP model loading and causal rollout.
        from model.dmd import DMD
        self.model = DMD(config, device=self.device)
        self.model.eval()
        if config.generator_ckpt and not config.resume:
            state = torch.load(config.generator_ckpt, map_location="cpu", weights_only=False)
            z_state = state.get("z_normalization")
            weights = state.get("generator", state.get("generator_ema", state.get("model", state)))
            weights = {name.replace("model._fsdp_wrapped_module.", "model.", 1): value
                       for name, value in weights.items()}
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            with FSDP.state_dict_type(self.model.generator, StateDictType.FULL_STATE_DICT,
                                      FullStateDictConfig(offload_to_cpu=True, rank0_only=False)):
                result = self.model.generator.load_state_dict(weights, strict=False)
            unexpected = result.unexpected_keys
            missing = [name for name in result.missing_keys if "lora" not in name.lower()]
            if unexpected or missing:
                raise ValueError(f"generator checkpoint incompatible: missing={missing[:5]}, unexpected={unexpected[:5]}")
            if z_state is not None:
                self.model.load_z_normalization_state_dict(z_state)
            del state, weights
        self.generator_optimizer = torch.optim.AdamW(
            [p for p in self.model.generator.parameters() if p.requires_grad],
            lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)
        self.fake_optimizer = torch.optim.AdamW(
            [p for p in self.model.fake_score.parameters() if p.requires_grad],
            lr=config.lr_critic, betas=(config.beta1_critic, config.beta2_critic), weight_decay=config.weight_decay)
        self.objective = Stage3Objective(self.model, config.image_or_video_shape,
            config.stage3_microbatch, dict(config.stage3_component_weights), config.stage3_normalize)
        from omegaconf import OmegaConf
        self.signature = OmegaConf.to_container(config, resolve=True)
        # Logging, run length and checkpoint paths may change when resuming.
        for key in ("resume", "max_train_steps", "log_iters", "logdir", "config_name",
                    "no_save", "no_visualize", "disable_wandb", "wandb_save_dir"):
            self.signature.pop(key, None)
        self.signature["dataset_sha256"] = self.dataset.fingerprint
        self.signature.pop("data_path", None)
        self.step, self.ema = 0, None
        self.output = Path(config.logdir or "stage3_outputs/training")
        self.output.mkdir(parents=True, exist_ok=True)
        if config.resume:
            self.step, self.ema = load_checkpoint(config.resume, self.model,
                self.generator_optimizer, self.fake_optimizer, self.stream, self.signature)
        if self.rank == 0:
            (self.output / "config.json").write_text(json.dumps(OmegaConf.to_container(config, resolve=True), indent=2), encoding="utf-8")
            print(f"Stage 3: {len(self.dataset)} valid captions, {self.world} ranks, step={self.step}", flush=True)

    def _subset(self):
        index = CONDITION_NAMES.index(sample_subset(dict(self.config.stage3_subset_probabilities))) if self.rank == 0 else 0
        value = torch.tensor([index], device=self.device)
        dist.broadcast(value, src=0)
        return CONDITION_NAMES[value.item()]

    def _clip(self, module, maximum):
        if hasattr(module, "clip_grad_norm_"):
            return module.clip_grad_norm_(maximum)
        return torch.nn.utils.clip_grad_norm_(module.parameters(), maximum)

    @torch.no_grad()
    def _update_ema(self):
        if not self.config.ema_weight or self.step < self.config.ema_start_step:
            return
        values = {name: p.detach().float().cpu() for name, p in self.model.generator.named_parameters() if p.requires_grad}
        if self.ema is None:
            self.ema = {name: value.clone() for name, value in values.items()}
        else:
            for name, value in values.items():
                self.ema[name].lerp_(value, 1 - self.config.ema_weight)

    def save(self):
        save_checkpoint(self.output / f"checkpoint_{self.step:07d}", self.model,
            self.generator_optimizer, self.fake_optimizer, self.stream,
            self.step, self.signature, self.ema)

    def train(self):
        last_saved = self.step
        while self.step < self.config.max_train_steps:
            started = time.monotonic()
            logs = {}
            if self.step % self.config.dfake_gen_update_ratio == 0:
                item = next(self.stream)
                self.generator_optimizer.zero_grad(set_to_none=True)
                with (torch.autograd.graph.save_on_cpu(pin_memory=True)
                    if getattr(self.config, "stage3_activation_cpu_offload", False) else nullcontext()):
                    loss, values = self.objective.generator_loss(item["conditions"])
                loss.backward()
                self.model.release_kv_cache()
                self._clip(self.model.generator, self.config.max_grad_norm_generator)
                self.generator_optimizer.step()
                self.generator_optimizer.zero_grad(set_to_none=True)
                self._update_ema()
                logs.update({key: value.item() for key, value in values.items()})
            item, subset = next(self.stream), self._subset()
            self.fake_optimizer.zero_grad(set_to_none=True)
            with (torch.autograd.graph.save_on_cpu(pin_memory=True)
                if getattr(self.config, "stage3_activation_cpu_offload", False) else nullcontext()):
                loss, values = self.objective.fake_loss(item["conditions"], subset)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite fake training loss")
            loss.backward()
            self.model.release_kv_cache()
            self._clip(self.model.fake_score, self.config.max_grad_norm_critic)
            self.fake_optimizer.step()
            self.fake_optimizer.zero_grad(set_to_none=True)
            self.step += 1
            logs.update({key: value.item() for key, value in values.items()})
            logs.update(step=self.step, subset=subset, seconds=time.monotonic() - started)
            if self.rank == 0:
                with (self.output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(logs) + "\n")
                print(json.dumps(logs), flush=True)
            if not self.config.no_save and self.step % self.config.log_iters == 0:
                self.save()
                last_saved = self.step
        if not self.config.no_save and self.step != last_saved:
            self.save()
