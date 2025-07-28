import torch
from typing import List, Optional, Union, Callable, Dict, Any
from diffusers import StableDiffusionPipeline
from dataclasses import dataclass
from diffusers.utils import BaseOutput
import PIL


@dataclass  
class StableDiffusionDualPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines that generate two images.
    
    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        nsfw_content_detected (`List[bool]`)
            List indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content or
            `None` if safety checking could not be performed.
    """
    images: Union[List[PIL.Image.Image], torch.Tensor]
    nsfw_content_detected: Optional[List[bool]]


class StableDiffusionDualPipeline(StableDiffusionPipeline):
    """
    Optimized pipeline for text-to-image generation using Stable Diffusion with dual prompt support.
    
    This pipeline extends the standard StableDiffusionPipeline to support generating two images
    from two different prompts (prompt_blend and prompt_fg) in a single forward pass by batching.
    """
    
    @torch.no_grad()
    def __call__(
        self,
        prompt_blend: Union[str, List[str]] = None,
        prompt_fg: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[Any] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], Any]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        """
        Function invoked when calling the pipeline for generation.
        
        Args:
            prompt_blend (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation for the blended image.
            prompt_fg (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation for the foreground image.
            ... (other parameters same as StableDiffusionPipeline)
            
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionDualPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionDualPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images.
        """
        
        # Ensure prompts are strings
        if isinstance(prompt_blend, list):
            prompt_blend = prompt_blend[0]
        if isinstance(prompt_fg, list):
            prompt_fg = prompt_fg[0]
            
        # Create batch of prompts
        batch_prompts = [prompt_blend, prompt_fg]
        
        # Handle negative prompts
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                batch_negative_prompts = [negative_prompt, negative_prompt]
            else:
                batch_negative_prompts = negative_prompt[:2]
        else:
            batch_negative_prompts = None
            
        # Generate both images in a single batch
        result = super().__call__(
            prompt=batch_prompts,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            negative_prompt=batch_negative_prompts,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_image_embeds=ip_adapter_image_embeds,
            output_type=output_type,
            return_dict=True,
            cross_attention_kwargs=cross_attention_kwargs,
            guidance_rescale=guidance_rescale,
            clip_skip=clip_skip,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            **kwargs,
        )
        
        # Split results
        if output_type == "pil":
            # Return list of images: [blend_image, fg_image]
            images = result.images[:2]  # Get first two images from batch
        else:
            # For tensor output, keep as is
            images = result.images
            
        if not return_dict:
            return (images, result.nsfw_content_detected)
            
        return StableDiffusionDualPipelineOutput(
            images=images,
            nsfw_content_detected=result.nsfw_content_detected
        )