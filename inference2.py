import argparse
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

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
    return parser.parse_args()

def main(args):
    target_path = args.output_dir
    
    img_path = os.path.join(target_path, 'image')
    fg_path = os.path.join(target_path, 'fg')
    
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

    pipe = DiffusionPipeline.from_pretrained(args.model_name, torch_dtype=torch.float32)
    
    scheduler_args = {}
    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type
        
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    pipe.load_lora_weights(args.lora_weights)
    
    pipe.set_progress_bar_config(disable=True)
        
    progress_bar = tqdm(range(cnt, args.num_images))
    progress_bar.set_description("Generating images")
        
    pipeline_args = {
        "prompt_blend": args.prompt_blend,
        "prompt_fg": args.prompt_fg,
        "num_inference_steps": args.num_inference_steps
    }
    
    
    for i in range(cnt, args.num_images):
        image_blend, image_fg = pipe(**pipeline_args, guidance_scale=2.5).images

        image_blend.save(os.path.join(img_path, f'{i}.png'))
        image_fg.save(os.path.join(fg_path, f'{i}.png'))
        
        progress_bar.update(1)

if __name__ == "__main__":
    args = parse_args()
    main(args)