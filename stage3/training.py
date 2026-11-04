"""Stage 3 objectives, shared by real Wan training and small smoke tests."""

import math
import torch

from .conditions import CONDITION_NAMES, CONDITION_INDICES
from .decomposition import decompose_conditions
from .fake_training import make_fake_score_batch, flow_target_loss
from .score_analysis import evaluate_conditions, make_wan_predictor


def validate_weights(weights):
    if set(weights) != set(CONDITION_NAMES) or not all(math.isfinite(float(v)) for v in weights.values()):
        raise ValueError("component weights must contain all eight names and finite values")
    if not any(float(value) != 0 for value in weights.values()):
        raise ValueError("at least one component weight must be nonzero")


def sigma_at(scheduler, timestep, reference):
    times = scheduler.timesteps.to(device=timestep.device, dtype=torch.float64)
    indices = (times[:, None] - timestep.double().reshape(1, -1)).abs().argmin(dim=0)
    sigma = scheduler.sigmas.to(device=reference.device, dtype=torch.float32)[indices]
    return sigma.reshape(timestep.shape)


class Stage3Objective:
    def __init__(self, model, shape, microbatch=1, weights=None, normalize=True):
        if len(shape) != 5 or shape[0] != 1:
            raise ValueError("Stage 3 currently requires batch_size=1 and [1,F,C,H,W] shape")
        if microbatch < 1:
            raise ValueError("condition microbatch must be positive")
        self.model, self.shape, self.microbatch = model, list(shape), microbatch
        self.weights = dict.fromkeys(CONDITION_NAMES, 1.) if weights is None else dict(weights)
        validate_weights(self.weights)
        self.normalize = normalize

    @torch.no_grad()
    def encode(self, texts):
        return self.model.text_encoder(text_prompts=texts)

    def rollout(self, conditions, subset):
        encoded = self.encode([conditions[subset]])
        result = self.model._run_generator(image_or_video_shape=self.shape, conditional_dict=encoded)
        return result[0], result[1]

    def noise_sample(self, rollout):
        # Discrete scheduler sampling avoids confusing timestep IDs with sigmas.
        scheduler = self.model.scheduler
        valid = ((scheduler.sigmas > 0.02) & (scheduler.sigmas < 0.98)).nonzero().flatten()
        if not len(valid):
            raise ValueError("scheduler has no training noise levels in (0.02, 0.98)")
        index = valid[torch.randint(len(valid), (1,), device=valid.device)]
        timestep = scheduler.timesteps[index].to(rollout.device).reshape(1, 1).expand(1, rollout.shape[1])
        sigma = sigma_at(scheduler, timestep, rollout)
        noise = torch.randn_like(rollout)
        expanded = sigma[..., None, None, None]
        noisy = (1 - expanded) * rollout.detach().float() + expanded * noise.float()
        return noisy.to(rollout.dtype), noise, timestep, sigma

    def generator_loss(self, conditions):
        rollout, mask = self.rollout(conditions, "PAB")
        with torch.no_grad():
            noisy, _, timestep, sigma = self.noise_sample(rollout)
            # All conditions and both models share this exact noisy sample/process.
            teacher = evaluate_conditions(noisy, timestep, conditions, self.encode,
                make_wan_predictor(self.model.real_score, "score", self.model.scheduler),
                microbatch=self.microbatch)
            fake = evaluate_conditions(noisy, timestep, conditions, self.encode,
                make_wan_predictor(self.model.fake_score, "score", self.model.scheduler),
                microbatch=self.microbatch)
            components = decompose_conditions(fake - teacher)
            score_difference = sum(self.weights[name] * components[name] for name in CONDITION_NAMES).unsqueeze(0)
            s = sigma[..., None, None, None]
            # x0_fake - x0_real = sigma^2/(1-sigma) * (score_fake-score_real).
            gradient = s.square() / (1 - s) * score_difference
            real_x0 = (noisy.float() + s.square() * teacher[CONDITION_INDICES["PAB"]].unsqueeze(0)) / (1 - s)
            if self.normalize:
                z = (rollout.detach().float() - real_x0).abs().mean((2, 3, 4))
                if hasattr(self.model, "_get_z_normalizer"):
                    z_current = z.mean(1, keepdim=True) if self.model.z_normalization_mode == "per_sample" else z
                    gradient /= self.model._get_z_normalizer(z_current, timestep).clamp_min(1e-6)
                    if self.model.z_normalization_mode == "timestep_ema":
                        self.model._update_timestep_z(z, timestep)
                else:
                    gradient /= z.mean().clamp_min(1e-6)
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("nonfinite Stage 3 score gradient")
        selected = torch.ones_like(rollout, dtype=torch.bool) if mask is None else mask.bool().expand_as(rollout)
        if not selected.any():
            raise ValueError("rollout gradient mask is empty")
        target = (rollout.float() - gradient).detach()
        loss = 0.5 * torch.nn.functional.mse_loss(rollout.float()[selected], target[selected])
        logs = {"generator_loss": loss.detach(), "gradient_norm": gradient.norm().detach()}
        logs.update({"component_" + name: value.norm().detach() for name, value in components.items()})
        return loss, logs

    def fake_loss(self, conditions, subset):
        with torch.no_grad():
            rollout, _ = self.rollout(conditions, subset)
            _, noise, timestep, sigma = self.noise_sample(rollout)
            batch = make_fake_score_batch(subset, conditions[subset], rollout.float(), noise.float(),
                1 - sigma, sigma, timestep, rollout_subset=subset, rollout_condition=conditions[subset])
            # Use the batch construction's noise process for both training and tests.
            encoded = self.encode([batch.condition])
        # The detached fake rollout no longer needs the generator's KV cache.
        if hasattr(self.model, "release_kv_cache"):
            self.model.release_kv_cache()
        flow, _ = self.model.fake_score(noisy_image_or_video=batch.noisy.to(rollout.dtype),
                                        conditional_dict=encoded, timestep=timestep)[:2]
        loss = flow_target_loss(flow.float(), batch)
        return loss, {"fake_loss": loss.detach()}
