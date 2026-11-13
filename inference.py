import argparse
import argparse
import torch
import os
import tempfile

import av
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
import json

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

from demo_utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21, help="Number of overlap frames between sliding windows")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--report_timing", action="store_true",
                    help="Only tested on A800, for the Causal Forcing++ latency. Not make claims for other hardware like H100. For the result on H100, refer to the reported results in the Self Forcing paper.")
parser.add_argument("--no_lora", action="store_true", help="Disable LoRA adapters for inference")
args = parser.parse_args()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
else:
    device = torch.device("cuda")
    rank = 0
    world_size = 1

set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
low_memory = get_cuda_free_memory_gb(device) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)
if args.checkpoint_path:
    companion = Path(args.checkpoint_path).with_suffix(".config.json")
    if companion.is_file():
        config = OmegaConf.merge(config, OmegaConf.load(companion))

if args.no_lora:
    OmegaConf.update(config, "model_kwargs.lora_rank", 0)

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

def load_generator_state(state):
    fixed = {}
    for name, value in state.items():
        name = name.replace("_fsdp_wrapped_module.", "")
        fixed[name] = value
    return fixed

base_path = config.get("generator_ckpt")
if base_path and (not args.checkpoint_path or Path(base_path).resolve() != Path(args.checkpoint_path).resolve()):
    base_checkpoint = torch.load(base_path, map_location="cpu")
    base_key = "generator_ema" if "generator_ema" in base_checkpoint else "generator"
    base_state = load_generator_state(base_checkpoint[base_key])
    pipeline.generator.load_state_dict(base_state, strict=False)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    key = "generator_ema" if args.use_ema else "generator"
    gen_sd = load_generator_state(state_dict[key])
    pipeline.generator.load_state_dict(gen_sd, strict=True)

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
else:
    pipeline.text_encoder.to(device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if args.report_timing and num_prompts < 2:
    print(f"[WARN] --report_timing requires at least 2 prompts "
          f"(got {num_prompts}); timing disabled.")
    args.report_timing = False

if dist.is_initialized():
    sampler = range(rank, num_prompts, world_size)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


def video_is_complete(path, expected_frames):
    if not os.path.isfile(path):
        return False
    try:
        with av.open(path) as container:
            if not container.streams.video:
                return False
            return sum(1 for _ in container.decode(video=0)) == expected_frames
    except (av.error.FFmpegError, OSError, EOFError):
        return False


def save_video(path, frames):
    with tempfile.NamedTemporaryFile(
        dir=os.path.dirname(path), prefix=".inference-", suffix=".mp4", delete=False
    ) as temporary:
        temporary_path = temporary.name
    try:
        write_video(temporary_path, frames, fps=16)
        os.replace(temporary_path, path)
    finally:
        Path(temporary_path).unlink(missing_ok=True)


expected_frames = 1 + 4 * (args.num_output_frames - 1)

for i, batch_data in tqdm(enumerate(dataloader), disable=(rank != 0)):
    idx = batch_data['idx'].item()

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames
    
    
    if args.i2v:
        assert config.num_frame_per_block == 1, "Current I2V only supports the frame-wise model."
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        output_path = os.path.join(args.output_folder, f'{prompt[:100]}.mp4')
        if video_is_complete(output_path, expected_frames):
            print('Video has been generated. Pass!')
            continue
        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        prompts = [prompt] 
        sampled_noise = torch.randn(
            [1, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
        output_path = os.path.join(args.output_folder, f'{prompt}.mp4')
        if video_is_complete(output_path, expected_frames):
            print('Video has been generated. Pass!')
            continue
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] 
        else:
            prompts = [prompt] 

        initial_latent = None
        sampled_noise = torch.randn(
            [1, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

    sample_report_timing = args.report_timing and i >= 1
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        report_timing=sample_report_timing,
    )
    if sample_report_timing:
        latency = pipeline.first_chunk_time
        elapsed = pipeline.last_generation_time
        num_pixel_frames = video.shape[1]
        fps = num_pixel_frames / elapsed if elapsed > 0 else float('inf')
        print(f"[Sample {i}] {num_pixel_frames} frames, "
              f"latency ↓ {latency:.2f}s, FPS ↑ {fps:.2f}")
        # Only tested on A800, for the Causal Forcing++ paper latency & throughput.
        # Not make claims for other hardware like H100.
        # For the result on H100, refer to the reported results in the Self Forcing paper.
        # We do not guarantee that our FPS/latency measurement protocol is identical to that used in the Self Forcing paper.
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    clean_latent = latents[0].cpu() 
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    save_video(output_path, video[0])

       

if dist.is_initialized():
    dist.destroy_process_group()
