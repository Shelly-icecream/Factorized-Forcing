"""Exercise the actual Stage 3 objectives and checkpoint code without Wan weights."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from .checkpoint import save_checkpoint, load_checkpoint
from .dataset import ParsedCaptionDataset, CaptionStream
from .parser import CaptionParser
from .training import Stage3Objective


class TinyScore(torch.nn.Module):
    uniform_timestep = True

    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([weight]))

    def forward(self, noisy_image_or_video, timestep, conditional_dict):
        flow = self.weight * noisy_image_or_video + conditional_dict["value"][:, None, None, None, None] * 0.01
        return flow, noisy_image_or_video - flow


class TinyRuntime(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = torch.nn.Linear(1, 1)
        self.real_score = TinyScore(0.7).requires_grad_(False)
        self.fake_score = TinyScore(0.2)
        self.scheduler = SimpleNamespace(timesteps=torch.tensor([900., 500., 100.]),
                                         sigmas=torch.tensor([0.9, 0.5, 0.1]))
        self.rollout_texts = []

    def text_encoder(self, text_prompts):
        device = next(self.generator.parameters()).device
        return {"value": torch.tensor([sum(map(ord, text)) % 31 + 1. for text in text_prompts], device=device)}

    def _run_generator(self, image_or_video_shape, conditional_dict):
        value = conditional_dict["value"].reshape(1, 1) / 31
        self.rollout_texts.append(value.item())
        rollout = self.generator(value).reshape(1, 1, 1, 1, 1).expand(image_or_video_shape)
        return rollout, torch.ones_like(rollout, dtype=torch.bool), None, None

    def release_kv_cache(self):
        pass


def run_smoke(use_fsdp=False):
    torch.manual_seed(1234)
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        parser = CaptionParser()
        captions = ["A man is walking in a park.", "A girl is running in a room.",
                    "A woman is walking on a beach."]
        source = root / "parsed.jsonl"
        source.write_text("".join(json.dumps(parser.parse(text).to_dict()) + "\n" for text in captions), encoding="utf-8")
        data = ParsedCaptionDataset(source)
        stream = CaptionStream(data, seed=42)
        model = TinyRuntime().eval()
        if use_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            model = model.cuda()
            model.generator = FSDP(model.generator, device_id=0, use_orig_params=True)
            model.fake_score = FSDP(model.fake_score, device_id=0, use_orig_params=True)
            model.real_score = FSDP(model.real_score, device_id=0, use_orig_params=True)
        objective = Stage3Objective(model, [1, 2, 1, 2, 2], microbatch=3)
        generator_optimizer = torch.optim.AdamW(model.generator.parameters(), lr=0.01)
        fake_optimizer = torch.optim.AdamW(model.fake_score.parameters(), lr=0.01)

        def step():
            generator_optimizer.zero_grad(set_to_none=True)
            loss, _ = objective.generator_loss(next(stream)["conditions"])
            loss.backward()
            if any(p.grad is not None for p in model.real_score.parameters()):
                raise AssertionError("teacher received gradients")
            if any(p.grad is not None for p in model.fake_score.parameters()):
                raise AssertionError("fake score received generator-loss gradients")
            generator_optimizer.step()
            generator_optimizer.zero_grad(set_to_none=True)
            fake_optimizer.zero_grad(set_to_none=True)
            item = next(stream)
            fake_loss, _ = objective.fake_loss(item["conditions"], "P")
            fake_loss.backward()
            if any(p.grad is not None for p in model.generator.parameters()):
                raise AssertionError("fake loss reached generator")
            fake_optimizer.step()
            fake_optimizer.zero_grad(set_to_none=True)
            return loss.item(), fake_loss.item()

        before_generator = [p.detach().clone() for p in model.generator.parameters()]
        before_fake = model.fake_score.weight.detach().clone()
        first = step()
        assert any(not torch.equal(before, after) for before, after in zip(before_generator, model.generator.parameters()))
        assert not torch.equal(before_fake, model.fake_score.weight)
        signature = {"dataset_sha256": data.fingerprint, "seed": 42}
        save_checkpoint(root / "checkpoint", model, generator_optimizer, fake_optimizer,
                        stream, 1, signature, {"test": torch.tensor(1.)})
        expected = step()
        expected_weights = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        restored_step, ema = load_checkpoint(root / "checkpoint", model, generator_optimizer,
                                             fake_optimizer, stream, signature)
        actual = step()
        assert restored_step == 1 and ema["test"].item() == 1
        assert expected == actual, (expected, actual)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, expected_weights[name], rtol=0, atol=0)
        return {"generator_loss": first[0], "fake_loss": first[1],
                "generator_updated": True, "fake_updated": True,
                "teacher_frozen": True, "resume_exact": True, "valid_captions": len(data)}


def run_fsdp_smoke():
    import torch.distributed as dist
    if not torch.cuda.is_available():
        raise RuntimeError("FSDP smoke test requires a CUDA GPU")
    if not dist.is_nccl_available():
        raise RuntimeError("CUDA FSDP smoke test requires an NCCL-enabled PyTorch build (run on the Linux training server)")
    with TemporaryDirectory() as temporary:
        store = (Path(temporary) / "store").as_uri()
        torch.cuda.set_device(0)
        dist.init_process_group("nccl", init_method=store, rank=0, world_size=1)
        try:
            result = run_smoke(use_fsdp=True)
            result["fsdp"] = True
            return result
        finally:
            dist.destroy_process_group()
