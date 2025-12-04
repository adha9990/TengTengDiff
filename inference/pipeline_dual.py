"""
Dual Stable Diffusion Pipeline for simultaneous anomaly image and mask generation.

Based on DualAnoDiff: https://github.com/yinyjin/DualAnoDiff
Paper: https://arxiv.org/abs/2408.13509
"""

from typing import Any, Callable, Dict, List, Optional, Union

import einops
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`.
    Based on https://arxiv.org/pdf/2305.08891.pdf Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class DualStableDiffusionPipeline(StableDiffusionPipeline):
    """
    Pipeline for dual text-to-image generation using Stable Diffusion.

    Simultaneously generates:
    - Anomaly image (blend): using prompt like "a vfx with sks"
    - Anomaly foreground (fg): using prompt like "sks"

    Both outputs share the same initial noise and diffusion process,
    ensuring spatial alignment between the generated image and mask region.
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt_blend: Union[str, List[str]] = None,
        prompt_fg: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        share_latents: bool = True,
    ):
        """
        Generate anomaly image and foreground mask simultaneously.

        Args:
            prompt_blend: Prompt for the full anomaly image (e.g., "a vfx with sks")
            prompt_fg: Prompt for the foreground/mask region (e.g., "sks")
            height: Image height (default: 512)
            width: Image width (default: 512)
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            negative_prompt: Negative prompt for generation
            num_images_per_prompt: Number of image pairs to generate per prompt
            eta: DDIM eta parameter
            generator: Random generator for reproducibility
            latents: Pre-generated latents
            output_type: Output format ("pil" or "np")
            return_dict: Whether to return a dict or tuple
            callback: Callback function during inference
            callback_steps: Callback frequency
            cross_attention_kwargs: Cross attention kwargs
            guidance_rescale: Guidance rescale factor
            share_latents: If True, blend and fg share the same initial noise (better alignment).
                          If False, use independent noise (original DualAnoDiff design).

        Returns:
            StableDiffusionPipelineOutput with images list containing:
            - Even indices (0, 2, 4, ...): Anomaly images (blend)
            - Odd indices (1, 3, 5, ...): Foreground regions (fg)
        """
        # 0. Default height and width
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(
            prompt_blend, height, width, callback_steps,
            negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt_blend is not None and isinstance(prompt_blend, str):
            batch_size = 1
        elif prompt_blend is not None and isinstance(prompt_blend, list):
            batch_size = len(prompt_blend)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompts (both blend and fg)
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None else None
        )

        # Encode blend prompt - returns (prompt_embeds, negative_prompt_embeds)
        prompt_embeds_blend, negative_prompt_embeds_blend = self.encode_prompt(
            prompt_blend,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # Encode fg prompt
        prompt_embeds_fg, negative_prompt_embeds_fg = self.encode_prompt(
            prompt_fg,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # Combine positive embeddings: [blend, fg] interleaved
        # prompt_embeds_blend: [batch, seq_len, hidden_dim]
        # prompt_embeds_fg: [batch, seq_len, hidden_dim]
        prompt_embeds_combined = torch.cat(
            [prompt_embeds_blend.unsqueeze(1), prompt_embeds_fg.unsqueeze(1)],
            dim=1
        )
        # Shape: [batch, 2, seq_len, hidden_dim] -> [batch*2, seq_len, hidden_dim]
        prompt_embeds_combined = einops.rearrange(
            prompt_embeds_combined, 'b f l c -> (b f) l c'
        )

        # Combine negative embeddings the same way
        if do_classifier_free_guidance:
            negative_prompt_embeds_combined = torch.cat(
                [negative_prompt_embeds_blend.unsqueeze(1),
                 negative_prompt_embeds_fg.unsqueeze(1)],
                dim=1
            )
            negative_prompt_embeds_combined = einops.rearrange(
                negative_prompt_embeds_combined, 'b f l c -> (b f) l c'
            )
            # For CFG: [negative, positive] in batch dimension
            prompt_embeds = torch.cat(
                [negative_prompt_embeds_combined, prompt_embeds_combined], dim=0
            )
        else:
            prompt_embeds = prompt_embeds_combined

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels

        if share_latents:
            # SHARED noise: Generate single latent and duplicate for blend and fg
            # This enforces spatial alignment between anomaly image and foreground mask
            latents_single = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
            latents = torch.cat([latents_single, latents_single], dim=0)
        else:
            # INDEPENDENT noise: Original DualAnoDiff design
            # Alignment relies on learned correlations during training
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt * 2,  # 2x for dual
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand latents for classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Predict noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # Perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                    )

                # Compute previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # Callback
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Decode latents
        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )
