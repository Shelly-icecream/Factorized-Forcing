"""CPU checks of the repository's actual distillation loop and P/A/B objective."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from model.factorized_objective import FactorizedObjective
from prompt_processing.parser import CaptionParser
from trainer.distillation import Trainer
from utils.distillation_checkpoint import load_checkpoint
from utils.distributed import EMA_FSDP
from utils.pab_conditions import TRAINING_CONDITION_NAMES as CONDITION_NAMES
from utils.pab_dataset import ParsedCaptionDataset, CaptionStream
from tests.factorized_smoke import TinyRuntime, run_smoke


class DistillationTrainingTests(unittest.TestCase):
    def make_trainer(self, root, source, steps=5):
        trainer = Trainer.__new__(Trainer)
        trainer.config = SimpleNamespace(
            max_train_steps=steps, dfake_gen_update_ratio=2, no_save=False,
            log_iters=2, ema_start_step=0, ema_weight=0.9, gc_interval=100,
            factorized_activation_cpu_offload=False,
        )
        trainer.factorized = True
        trainer.step = 0
        trainer.device = torch.device("cpu")
        trainer.is_main_process = True
        trainer.disable_wandb = True
        trainer.previous_time = None
        trainer.output_path = str(root)
        root.mkdir(parents=True)
        trainer.model = TinyRuntime().eval()
        trainer.dataset = ParsedCaptionDataset(source)
        trainer.dataloader = CaptionStream(trainer.dataset, seed=42)
        trainer.factorized_objective = FactorizedObjective(trainer.model, [1, 2, 1, 2, 2], microbatch=3)
        trainer.generator_optimizer = torch.optim.AdamW(trainer.model.generator.parameters(), lr=0.01)
        trainer.critic_optimizer = torch.optim.AdamW(trainer.model.fake_score.parameters(), lr=0.01)
        trainer.max_grad_norm_generator = trainer.max_grad_norm_critic = 10.
        trainer.generator_ema = EMA_FSDP(trainer.model.generator, decay=0.9, trainable_only=True)
        trainer.checkpoint_signature = {"dataset_sha256": trainer.dataset.fingerprint, "seed": 42}
        return trainer

    def test_objective_never_encodes_singletons_and_rejects_their_rollouts(self):
        model = TinyRuntime().eval()
        objective = FactorizedObjective(model, [1, 2, 1, 2, 2], microbatch=2)
        conditions = {name: name for name in ("empty", "P", "A", "B", "PA", "PB", "AB", "PAB")}
        with patch.object(model, "text_encoder", wraps=model.text_encoder) as encoder:
            loss, _ = objective.generator_loss(conditions)
            loss.backward()
            with patch.object(model, "critic_loss", wraps=model.critic_loss) as critic:
                with patch.object(model, "_run_generator", wraps=model._run_generator) as rollout:
                    with patch.object(model.fake_score, "forward", wraps=model.fake_score.forward) as predict:
                        for parameter in model.generator.parameters():
                            parameter.grad = None
                        fake_loss, _ = objective.fake_loss(conditions)
                        fake_loss.backward()
                        critic.assert_called_once()
                        expected = model.text_encoder(text_prompts=["PAB"])
                        torch.testing.assert_close(rollout.call_args.args[1]["value"], expected["value"])
                        torch.testing.assert_close(predict.call_args.kwargs["conditional_dict"]["value"],
                                                   expected["value"])
                        self.assertTrue(all(p.grad is None for p in model.generator.parameters()))
                        self.assertTrue(all(p.grad is None for p in model.real_score.parameters()))
            texts = [text for call in encoder.call_args_list for text in call.kwargs["text_prompts"]]
            self.assertTrue(set(texts) <= set(CONDITION_NAMES))
            self.assertEqual(texts[-1], "PAB")

    def test_shared_loop_updates_both_models_and_resumes_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "parsed.jsonl"
            parser = CaptionParser()
            source.write_text("".join(json.dumps(parser.parse(c).to_dict()) + "\n" for c in (
                "A man is walking in a park.", "A girl is running in a room.")), encoding="utf-8")
            torch.manual_seed(42)
            trainer = self.make_trainer(root / "full", source)
            before = {name: value.clone() for name, value in trainer.model.state_dict().items()}
            with patch("trainer.distillation.dist.broadcast"), patch("torch.cuda.empty_cache"):
                trainer.train()
            self.assertEqual(trainer.step, 5)
            self.assertFalse(torch.equal(before['generator.weight'], trainer.model.generator.weight))
            self.assertFalse(torch.equal(before['fake_score.weight'], trainer.model.fake_score.weight))
            torch.testing.assert_close(before['real_score.weight'], trainer.model.real_score.weight)
            self.assertTrue(all(p.grad is None for p in trainer.model.real_score.parameters()))
            self.assertEqual(len(trainer.model.rollout_texts), 8)  # 3 generator + 5 critic rollouts
            for step in (2, 4, 5):
                self.assertTrue((root / "full" / f"checkpoint_{step:07d}" / "metadata.json").exists())
            expected = [json.loads(line) for line in (root / "full" / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(sum('generator_loss' in item for item in expected), 3)
            self.assertTrue(all('critic_loss' in item for item in expected))

            restored = self.make_trainer(root / "resumed", source)
            restored.step, ema = load_checkpoint(root / "full" / "checkpoint_0000002", restored.model,
                restored.generator_optimizer, restored.critic_optimizer, restored.dataloader,
                restored.checkpoint_signature)
            restored.generator_ema.load_state_dict(ema)
            with patch("trainer.distillation.dist.broadcast"), patch("torch.cuda.empty_cache"):
                restored.train()
            actual = [json.loads(line) for line in (root / "resumed" / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(expected[2:], actual)
            for name, value in trainer.model.state_dict().items():
                torch.testing.assert_close(value, restored.model.state_dict()[name], rtol=0, atol=0)
            for name, value in trainer.generator_ema.state_dict().items():
                torch.testing.assert_close(value, restored.generator_ema.state_dict()[name], rtol=0, atol=0)

    def test_official_trainer_initializes_and_restores_factorized_runtime(self):
        from omegaconf import OmegaConf
        import model
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "captions.jsonl"
            parser = CaptionParser()
            source.write_text(json.dumps(parser.parse("A man is walking in a park.").to_dict()) + "\n")
            config = OmegaConf.merge(
                OmegaConf.load("configs/default_config.yaml"),
                OmegaConf.load("configs/factorized_dmd_framewise.yaml"),
                {"data_path": str(source), "factorized_parser_version": parser.parser_version,
                 "generator_ckpt": None, "resume": None, "logdir": str(root / "first"),
                 "image_or_video_shape": [1, 2, 1, 2, 2], "max_train_steps": 2,
                 "factorized_microbatch": 3, "disable_wandb": True, "no_visualize": True,
                 "no_save": False, "ema_start_step": 0, "ema_weight": 0.9},
            )
            def gather(output, value):
                output[0] = value
            with patch.dict(model.__dict__, {"DMD": lambda *args, **kwargs: TinyRuntime()}), \
                 patch("trainer.distillation.launch_distributed_job"), \
                 patch("trainer.distillation.dist.get_rank", return_value=0), \
                 patch("trainer.distillation.dist.get_world_size", return_value=1), \
                 patch("trainer.distillation.dist.all_gather_object", side_effect=gather), \
                 patch("trainer.distillation.dist.barrier"), \
                 patch("trainer.distillation.dist.broadcast"), \
                 patch("torch.cuda.current_device", return_value=torch.device("cpu")), \
                 patch("torch.cuda.empty_cache"):
                first = Trainer(config)
                self.assertIsInstance(first.factorized_objective, FactorizedObjective)
                first.train()
                config.resume = str(root / "first" / "checkpoint_0000002")
                config.max_train_steps = 3
                config.logdir = str(root / "restored")
                restored = Trainer(config)
                self.assertEqual(restored.step, 2)
                for name, value in first.model.state_dict().items():
                    torch.testing.assert_close(value, restored.model.state_dict()[name], rtol=0, atol=0)
                restored.train()
                self.assertEqual(restored.step, 3)
                self.assertTrue((root / "restored" / "checkpoint_0000003" / "metadata.json").exists())

    def test_objective_gradient_isolation_and_checkpoint(self):
        self.assertTrue(run_smoke()["resume_exact"])

    def test_standard_dmd_forward_backward_still_uses_original_losses(self):
        class BaselineRuntime(TinyRuntime):
            def generator_loss(self, **kwargs):
                self.generator_kwargs = kwargs
                return self.generator.weight.square().sum(), {"dmdtrain_gradient_norm": torch.tensor(1.)}

            def critic_loss(self, **kwargs):
                self.critic_kwargs = kwargs
                return self.fake_score.weight.square().sum(), {}

        trainer = Trainer.__new__(Trainer)
        trainer.factorized = False
        trainer.step = 1
        trainer.config = SimpleNamespace(i2v=False, image_or_video_shape=[1, 2, 1, 2, 2], negative_prompt="bad")
        trainer.model = BaselineRuntime()
        trainer.max_grad_norm_generator = trainer.max_grad_norm_critic = 10.
        batch = {"prompts": ["A man walks in a park."]}
        with patch("torch.cuda.synchronize"), patch("trainer.distillation.dist.get_rank", return_value=0):
            generator = trainer.fwdbwd_one_step(batch, True)
            critic = trainer.fwdbwd_one_step(batch, False)
        self.assertIn('generator_loss', generator)
        self.assertIn('critic_loss', critic)
        self.assertIs(trainer.model.generator_kwargs['unconditional_dict'], trainer.unconditional_dict)
        self.assertIs(trainer.model.critic_kwargs['unconditional_dict'], trainer.unconditional_dict)

    def test_trainable_ema_does_not_copy_frozen_base_weights(self):
        module = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Linear(1, 1))
        module[0].requires_grad_(False)
        ema = EMA_FSDP(module, trainable_only=True)
        self.assertEqual(set(ema.state_dict()), {'1.weight', '1.bias'})
        ema.update(module)
        full = EMA_FSDP(module)
        self.assertEqual(set(full.state_dict()), set(dict(module.named_parameters())))


if __name__ == '__main__':
    unittest.main()
