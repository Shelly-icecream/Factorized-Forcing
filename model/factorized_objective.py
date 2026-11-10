"""Factorized DMD objectives, shared by real Wan training and small smoke tests."""

import math
from itertools import combinations
import torch

from utils.pab_conditions import TRAINING_CONDITION_NAMES as CONDITION_NAMES, COMPONENT_NAMES
from model.score_decomposition import decompose_training_conditions
from model.condition_scores import evaluate_conditions, make_wan_predictor


def validate_weights(weights):
    if set(weights) != set(COMPONENT_NAMES) or not all(math.isfinite(float(v)) for v in weights.values()):
        raise ValueError("component weights must contain only empty/P/B/A/PAB and finite values")
    if not any(float(value) != 0 for value in weights.values()):
        raise ValueError("at least one component weight must be nonzero")


def sigma_at(scheduler, timestep, reference):
    times = scheduler.timesteps.to(device=timestep.device, dtype=torch.float64)
    indices = (times[:, None] - timestep.double().reshape(1, -1)).abs().argmin(dim=0)
    sigma = scheduler.sigmas.to(device=reference.device, dtype=torch.float32)[indices]
    return sigma.reshape(timestep.shape)


@torch.no_grad()
def score_direction_metrics(terms, full_conditional, baseline_cfg, mask=None):
    """Local-rank score-space diagnostics before sigma scaling/normalization.

    Zero-vector cosines/ratios are reported as zero with a validity flag.
    A low sum_to_norm_sum indicates cancellation, not necessarily bad quality.
    """
    def vector(value):
        value = value.detach().float()
        return value.reshape(-1) if mask is None else value[mask.bool().expand_as(value)]

    vectors = {name: vector(value) for name, value in terms.items()}
    refs = {"full_conditional": vector(full_conditional), "baseline_cfg": vector(baseline_cfg)}
    norms = {name: value.norm() for name, value in vectors.items()}
    logs = {}

    def cosine(name, left, right):
        denominator = left.norm() * right.norm()
        valid = denominator > 1e-12
        logs[name] = torch.where(valid, torch.dot(left, right) / denominator.clamp_min(1e-12), 0.)
        logs[name + "_valid"] = valid.float()

    for ref, value in refs.items():
        logs[f"score/reference_norm/{ref}"] = value.norm()
    for name, value in vectors.items():
        logs[f"score/weighted_norm/{name}"] = norms[name]
        for ref, reference in refs.items():
            denominator = reference.norm()
            key = f"score/relative_norm/{name}_to_{ref}"
            logs[key] = torch.where(denominator > 1e-12, norms[name] / denominator.clamp_min(1e-12), 0.)
            logs[key + "_valid"] = (denominator > 1e-12).float()
            cosine(f"score/cosine/{name}_vs_{ref}", value, reference)
    for left, right in combinations(vectors, 2):
        cosine(f"score/cosine/{left}_vs_{right}", vectors[left], vectors[right])
    total = sum(vectors.values())
    norm_sum = sum(norms.values())
    logs["score/total_norm"] = total.norm()
    logs["score/sum_to_norm_sum"] = torch.where(norm_sum > 1e-12, total.norm() / norm_sum.clamp_min(1e-12), 0.)
    logs["score/sum_to_norm_sum_valid"] = (norm_sum > 1e-12).float()
    return logs


class FactorizedObjective:
    def __init__(self, model, shape, microbatch=1, weights=None, normalize=True):
        if len(shape) != 5 or shape[0] != 1:
            raise ValueError("Factorized DMD currently requires batch_size=1 and [1,F,C,H,W] shape")
        if microbatch < 1:
            raise ValueError("condition microbatch must be positive")
        self.model, self.shape, self.microbatch = model, list(shape), microbatch
        self.weights = dict.fromkeys(COMPONENT_NAMES, 1.) if weights is None else dict(weights)
        validate_weights(self.weights)
        self.normalize = normalize

    @torch.no_grad()
    def encode(self, texts):
        return self.model.text_encoder(text_prompts=texts)

    def rollout(self, conditions, subset):
        if subset not in CONDITION_NAMES:
            raise ValueError("singleton P/A/B conditions are disabled for training")
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
                microbatch=self.microbatch, condition_names=CONDITION_NAMES)
            fake = evaluate_conditions(noisy, timestep, conditions, self.encode,
                make_wan_predictor(self.model.fake_score, "score", self.model.scheduler),
                microbatch=self.microbatch, condition_names=CONDITION_NAMES)
            components = decompose_training_conditions(fake - teacher)
            score_difference = sum(self.weights[name] * components[name] for name in COMPONENT_NAMES).unsqueeze(0)
            # Preserve baseline CFG separately from the factor contrasts. In
            # particular, R (the "empty" condition) is not the negative prompt.
            full = CONDITION_NAMES.index("PAB")
            teacher_full = teacher[full].unsqueeze(0)
            cfg_correction = torch.zeros_like(teacher_full)
            real_scale = getattr(self.model, "real_guidance_scale", 0.0)
            fake_scale = getattr(self.model, "fake_guidance_scale", 0.0)
            if real_scale != 0.0 or fake_scale != 0.0:
                negative = self.encode([self.model.args.negative_prompt])
                if real_scale != 0.0:
                    teacher_negative = make_wan_predictor(
                        self.model.real_score, "score", self.model.scheduler)(noisy, timestep, negative, None)
                    teacher_guidance = real_scale * (teacher_full - teacher_negative)
                    score_difference = score_difference - teacher_guidance
                    cfg_correction = cfg_correction - teacher_guidance
                    teacher_full = teacher_full + teacher_guidance
                if fake_scale != 0.0:
                    fake_negative = make_wan_predictor(
                        self.model.fake_score, "score", self.model.scheduler)(noisy, timestep, negative, None)
                    fake_guidance = fake_scale * (fake[full].unsqueeze(0) - fake_negative)
                    score_difference = score_difference + fake_guidance
                    cfg_correction = cfg_correction + fake_guidance
            full_difference = (fake[full] - teacher[full]).unsqueeze(0)
            terms = {name: (self.weights[name] * value).unsqueeze(0) for name, value in components.items()}
            terms["CFG"] = cfg_correction
            direction_logs = score_direction_metrics(
                terms, full_difference, full_difference + cfg_correction, mask)
            s = sigma[..., None, None, None]
            # x0_fake - x0_real = sigma^2/(1-sigma) * (score_fake-score_real).
            gradient = s.square() / (1 - s) * score_difference
            real_x0 = (noisy.float() + s.square() * teacher_full) / (1 - s)
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
                raise FloatingPointError("nonfinite factorized score gradient")
        selected = torch.ones_like(rollout, dtype=torch.bool) if mask is None else mask.bool().expand_as(rollout)
        if not selected.any():
            raise ValueError("rollout gradient mask is empty")
        target = (rollout.float() - gradient).detach()
        loss = 0.5 * torch.nn.functional.mse_loss(rollout.float()[selected], target[selected])
        logs = {"generator_loss": loss.detach(), "gradient_norm": gradient.norm().detach()}
        logs.update({"component_" + name: value.norm().detach() for name, value in components.items()})
        logs.update(direction_logs)
        logs["score/sigma_mean"] = sigma.float().mean()
        logs["score/timestep_mean"] = timestep.float().mean()
        return loss, logs

    def fake_loss(self, conditions):
        # Use the original Causal Forcing critic objective unchanged.
        conditional = self.encode([conditions["PAB"]])
        return self.model.critic_loss(
            image_or_video_shape=self.shape,
            conditional_dict=conditional,
            unconditional_dict=None,  # DMD.critic_loss does not use this argument.
            clean_latent=None,
        )


def validate_factorized_config(config):
    """Check the optional P/A/B objective before model memory is allocated."""
    if config.batch_size != 1 or list(config.image_or_video_shape)[0] != 1:
        raise ValueError("factorized DMD requires per-rank batch_size=1")
    if config.i2v:
        raise ValueError("factorized DMD currently supports text-to-video only")
    if not config.train_fake_score or not config.staged_fsdp_init:
        raise ValueError("factorized DMD requires train_fake_score and staged_fsdp_init")
    if config.denoising_loss_type != "flow":
        raise ValueError("factorized fake training requires denoising_loss_type=flow")
    if config.factorized_microbatch < 1:
        raise ValueError("factorized_microbatch must be positive")
    if config.max_train_steps < 1 or config.dfake_gen_update_ratio < 1 or config.log_iters < 1:
        raise ValueError("step counts, update ratio and save interval must be positive")
    if not 0 <= config.ema_weight < 1:
        raise ValueError("ema_weight must be in [0, 1)")
    validate_weights(dict(config.factorized_component_weights))
