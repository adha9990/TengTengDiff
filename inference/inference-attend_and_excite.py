import argparse
from pipeline_attend_and_excite import AttendAndExcitePipeline
from diffusers import DPMSolverMultistepScheduler
from utils.ptp_utils import AttentionStore, register_attention_control

import torch
import os
from tqdm.auto import tqdm
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lora_weights", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--prompt", type=str, default="a vfx")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--output_dir", type=str, required=True)

    # Attend-and-Excite 特定參數
    parser.add_argument(
        "--token_indices",
        type=str,
        default=None,
        help="Comma-separated token indices to alter (e.g., '2,5'). If None, no attention alteration is applied.",
    )
    parser.add_argument(
        "--max_iter_to_alter",
        type=int,
        default=25,
        help="Maximum denoising steps to apply Attend-and-Excite (default: 25)",
    )
    parser.add_argument(
        "--scale_factor",
        type=int,
        default=20,
        help="Scale factor for latent update (default: 20)",
    )
    parser.add_argument(
        "--attention_res",
        type=int,
        default=16,
        help="Resolution of attention maps from UNet (default: 16)",
    )
    parser.add_argument(
        "--smooth_attentions",
        action="store_true",
        help="Apply Gaussian smoothing to attention maps",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help="Standard deviation for Gaussian smoothing (default: 0.5)",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=3,
        help="Kernel size for Gaussian smoothing (default: 3)",
    )

    # 記憶體和性能優化參數
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
        "--seed", type=int, default=None, help="Random seed for reproducible generation"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for classifier-free guidance (default: 7.5)",
    )

    return parser.parse_args()


def parse_token_indices(token_indices_str):
    """Parse comma-separated token indices string to list of integers."""
    if token_indices_str is None or token_indices_str.strip() == "":
        return None
    try:
        indices = [int(idx.strip()) for idx in token_indices_str.split(",")]
        return indices
    except ValueError:
        logging.error(f"Invalid token indices format: {token_indices_str}")
        return None


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

    # 解析 token indices
    token_indices = parse_token_indices(args.token_indices)
    if token_indices is not None:
        logging.info(f"Using Attend-and-Excite with token indices: {token_indices}")
    else:
        logging.info("No token indices specified, using standard pipeline")

    # 選擇資料類型
    dtype = torch.float16 if args.use_fp16 else torch.float32

    # 使用 AttendAndExcitePipeline 替代標準 pipeline
    pipe = AttendAndExcitePipeline.from_pretrained(
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

    # 如果使用 Attend-and-Excite，創建並註冊 AttentionStore
    attention_store = None
    if token_indices is not None:
        attention_store = AttentionStore()
        register_attention_control(pipe, attention_store)
        logging.info("Registered attention control for Attend-and-Excite")

    progress_bar = tqdm(range(cnt, args.num_images))
    progress_bar.set_description("Generating images with Attend-and-Excite")

    # 基本 pipeline 參數
    pipeline_args = {
        "prompt": args.prompt,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
    }

    # 如果指定了 token_indices，添加 Attend-and-Excite 參數
    if token_indices is not None:
        # 為漸進式閾值設定（參考 AnomalyAny 的設定）
        thresholds = {
            0: 0.05,   # 初期：低閾值
            10: 0.5,   # 中期：中等閾值
            20: 0.8    # 後期：高閾值
        }

        pipeline_args.update({
            "indices_to_alter": token_indices,
            "max_iter_to_alter": args.max_iter_to_alter,
            "thresholds": thresholds,
            "scale_factor": args.scale_factor,
            "attention_res": args.attention_res,
            "smooth_attentions": args.smooth_attentions,
            "sigma": args.sigma,
            "kernel_size": args.kernel_size,
        })

        logging.info(f"Attend-and-Excite parameters:")
        logging.info(f"  - Token indices: {token_indices}")
        logging.info(f"  - Max iter to alter: {args.max_iter_to_alter}")
        logging.info(f"  - Scale factor: {args.scale_factor}")
        logging.info(f"  - Thresholds: {thresholds}")
        logging.info(f"  - Attention resolution: {args.attention_res}")
        logging.info(f"  - Smooth attentions: {args.smooth_attentions}")

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

        # 如果使用 Attend-and-Excite，重置並添加 AttentionStore 到參數
        if token_indices is not None:
            attention_store.reset()
            pipeline_args["attention_store"] = attention_store

        result = pipe(**pipeline_args, generator=generator).images[0]

        # 保存圖片
        result.save(os.path.join(img_path, f"{i}.png"))

        logging.info(f"Generated and saved image {i}")
        progress_bar.update(1)

    logging.info(f"Successfully generated {args.num_images - cnt} images")


if __name__ == "__main__":
    args = parse_args()
    main(args)
