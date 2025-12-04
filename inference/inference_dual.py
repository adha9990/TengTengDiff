"""
Dual inference script for simultaneous anomaly image and mask generation.

Based on DualAnoDiff: https://github.com/yinyjin/DualAnoDiff
Paper: https://arxiv.org/abs/2408.13509

Usage:
    python inference_dual.py \
        --model_name models/stable-diffusion-v1-5 \
        --lora_weights all_generate/hazelnut/hole/checkpoint-5000 \
        --output_dir generate_data/hazelnut/hole \
        --num_images 100 \
        --prompt_blend "a vfx with sks" \
        --prompt_fg "sks"
"""

import argparse
import logging
import os

import torch
from diffusers import DPMSolverMultistepScheduler
from tqdm.auto import tqdm

from pipeline_dual import DualStableDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate anomaly images and masks simultaneously"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Path to pretrained Stable Diffusion model"
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        required=True,
        help="Path to LoRA weights checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=100,
        help="Number of image pairs to generate"
    )
    parser.add_argument(
        "--prompt_blend",
        type=str,
        default="a vfx with sks",
        help="Prompt for full anomaly image"
    )
    parser.add_argument(
        "--prompt_fg",
        type=str,
        default="sks",
        help="Prompt for foreground/mask region"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt for generation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use FP16 mixed precision"
    )
    parser.add_argument(
        "--enable_xformers",
        action="store_true",
        help="Enable xformers for memory efficiency"
    )
    parser.add_argument(
        "--enable_vae_slicing",
        action="store_true",
        help="Enable VAE slicing for memory efficiency"
    )
    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        help="Enable model CPU offload"
    )
    parser.add_argument(
        "--share_latents",
        action="store_true",
        default=True,
        help="Share initial latent noise between blend and fg for better alignment"
    )
    parser.add_argument(
        "--no_share_latents",
        action="store_true",
        help="Use independent latent noise (original DualAnoDiff design)"
    )
    return parser.parse_args()


def main(args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Set random seed to {args.seed}")

    # Create output directories
    img_path = os.path.join(args.output_dir, "image")
    fg_path = os.path.join(args.output_dir, "fg")

    os.makedirs(img_path, exist_ok=True)
    os.makedirs(fg_path, exist_ok=True)

    # Count existing images to resume generation
    cnt = len(os.listdir(img_path))
    if cnt > 0:
        logger.info(f"Found {cnt} existing images, resuming from {cnt}")

    # Select dtype
    dtype = torch.float16 if args.use_fp16 else torch.float32

    # Load pipeline
    logger.info(f"Loading model from {args.model_name}")
    pipe = DualStableDiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # Use DPM-Solver for faster inference
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Load LoRA weights
    logger.info(f"Loading LoRA weights from {args.lora_weights}")
    pipe.load_lora_weights(args.lora_weights)

    # Enable memory optimizations
    if args.enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")

    if args.enable_vae_slicing:
        pipe.enable_vae_slicing()
        logger.info("Enabled VAE slicing")

    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
        logger.info("Enabled model CPU offload")
    else:
        pipe = pipe.to("cuda")

    pipe.set_progress_bar_config(disable=True)

    # Generate images
    logger.info(f"Generating {args.num_images - cnt} image pairs...")
    # Determine share_latents setting
    share_latents = not args.no_share_latents

    logger.info(f"  Prompt (blend): {args.prompt_blend}")
    logger.info(f"  Prompt (fg): {args.prompt_fg}")
    logger.info(f"  Guidance scale: {args.guidance_scale}")
    logger.info(f"  Inference steps: {args.num_inference_steps}")
    logger.info(f"  Share latents: {share_latents}")

    progress_bar = tqdm(range(cnt, args.num_images), desc="Generating image pairs")

    # Create generator for reproducibility
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cuda")

    for i in progress_bar:
        # Set seed for this iteration
        if generator is not None:
            generator.manual_seed(args.seed + i)

        # Generate dual images
        result = pipe(
            prompt_blend=args.prompt_blend,
            prompt_fg=args.prompt_fg,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
            generator=generator,
            share_latents=share_latents,
        )

        # Save images
        # result.images[0] = blend (full anomaly image)
        # result.images[1] = fg (foreground/mask region)
        result.images[0].save(os.path.join(img_path, f"{i}.png"))
        result.images[1].save(os.path.join(fg_path, f"{i}.png"))

        progress_bar.set_postfix({"saved": f"{i}.png"})

    logger.info(f"Generation complete! Images saved to {args.output_dir}")
    logger.info(f"  Anomaly images: {img_path}")
    logger.info(f"  Foreground regions: {fg_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
