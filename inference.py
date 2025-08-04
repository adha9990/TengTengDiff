import argparse
from diffusers import DPMSolverMultistepScheduler
from pipeline_stable_diffusion_dual import StableDiffusionDualPipeline

import torch
import os
from tqdm.auto import tqdm
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lora_weights", type=str, required=True)
    parser.add_argument("--mvtec_name", type=str, default="hazelnut")
    parser.add_argument("--mvtec_aomaly_name", type=str, default="hole")
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--prompt_blend", type=str, default="a vfx with sks")
    parser.add_argument("--prompt_fg", type=str, default="sks")
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--enable_xformers",
        action="store_true",
        help="Enable xformers for memory efficiency",
    )
    parser.add_argument(
        "--use_fp16", action="store_true", help="Use FP16 mixed precision"
    )
    parser.add_argument(
        "--enable_cpu_offload",
        action="store_true",
        help="Enable CPU offload for memory savings (very slow)",
    )
    parser.add_argument(
        "--enable_vae_slicing",
        action="store_true",
        help="Enable VAE slicing for memory efficiency",
    )
    parser.add_argument(
        "--enable_vae_tiling",
        action="store_true",
        help="Enable VAE tiling for huge image generation",
    )
    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        help="Enable model CPU offload (faster than sequential)",
    )
    parser.add_argument(
        "--disable_safety_checker",
        action="store_true",
        help="Disable NSFW safety checker",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducible generation"
    )
    return parser.parse_args()


def main(args):
    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logging.info(f"Set random seed to {args.seed}")

    target_path = args.output_dir

    img_path = os.path.join(target_path, "image")
    fg_path = os.path.join(target_path, "fg")

    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(fg_path):
        os.mkdir(fg_path)
    cnt = len(os.listdir(img_path))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # 選擇資料類型
    dtype = torch.float16 if args.use_fp16 else torch.float32
    
    pipe = StableDiffusionDualPipeline.from_pretrained(
        args.model_name, torch_dtype=dtype
    )

    scheduler_args = {}
    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, **scheduler_args
    )

    # 載入 LoRA 權重
    pipe.load_lora_weights(args.lora_weights)
    
    # 禁用安全檢查器（如果需要）
    if args.disable_safety_checker:
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        logging.info("Disabled NSFW safety checker")

    # 啟用記憶體優化選項
    if args.enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logging.info("Enabled xformers memory efficient attention")
        except Exception as e:
            logging.warning(f"Could not enable xformers: {e}")
    
    # 啟用 VAE 優化
    if args.enable_vae_slicing:
        pipe.enable_vae_slicing()
        logging.info("Enabled VAE slicing for memory efficiency")
    
    if args.enable_vae_tiling:
        pipe.enable_vae_tiling()
        logging.info("Enabled VAE tiling for huge image generation")

    # 設定 GPU 或 CPU offload
    if args.enable_cpu_offload:
        pipe.enable_sequential_cpu_offload()
        logging.info("Enabled sequential CPU offload (very slow)")
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
        logging.info("Enabled model CPU offload")
    else:
        pipe = pipe.to("cuda")

    pipe.set_progress_bar_config(disable=True)

    progress_bar = tqdm(range(cnt, args.num_images))
    progress_bar.set_description("Generating images")

    pipeline_args = {
        "prompt_blend": args.prompt_blend,
        "prompt_fg": args.prompt_fg,
        "num_inference_steps": args.num_inference_steps,
    }

    # Create generator for seed if specified
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(args.seed)

    for i in range(cnt, args.num_images):
        # Use the same seed for each generation if specified
        if generator is not None:
            generator.manual_seed(
                args.seed + i
            )  # Add i to get different images with predictable seeds
        
        # 使用雙重管線生成圖片和遮罩
        result_blend, result_fg = pipe(
            prompt_blend=args.prompt_blend,
            prompt_fg=args.prompt_fg,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=2.5,
            generator=generator,
            height=512,
            width=512,
        )
        
        # 獲取生成的圖片
        image_blend = result_blend.images[0]
        image_fg = result_fg.images[0]
        
        # 保存圖片
        image_blend.save(os.path.join(img_path, f"{i}.png"))
        image_fg.save(os.path.join(fg_path, f"{i}.png"))
        
        logging.info(f"Generated and saved image pair {i}")
        progress_bar.update(1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
