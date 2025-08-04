import copy
import gc
import logging
import math
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from ..data import DreamBoothDataset, PromptDataset, collate_fn
from ..models.utils import import_model_class_from_model_name_or_path, encode_prompt
from ..utils.tokenization import tokenize_prompt
from ..utils.model_card import save_model_card
from .validation import log_validation

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.35.0.dev0")

logger = get_logger(__name__)


class DreamBoothLoRATrainer:
    def __init__(self, args):
        self.args = args
        self.accelerator = None

    def setup_accelerator(self):
        """Initialize the accelerator"""
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)

        accelerator_project_config = ProjectConfiguration(
            project_dir=self.args.output_dir, logging_dir=logging_dir
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            project_config=accelerator_project_config,
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            self.accelerator.native_amp = False

        return self.accelerator

    def validate_args(self):
        """Validate training arguments"""
        if self.args.report_to == "wandb" and self.args.hub_token is not None:
            raise ValueError(
                "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
                " Please use `hf auth login` to authenticate with the Hub."
            )

        if self.args.report_to == "wandb":
            if not is_wandb_available():
                raise ImportError(
                    "Make sure to install wandb if you want to use it for logging during training."
                )

        # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
        if (
            self.args.train_text_encoder
            and self.args.gradient_accumulation_steps > 1
            and self.accelerator.num_processes > 1
        ):
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )

    def setup_logging(self):
        """Setup logging configuration"""
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

    def generate_class_images(self):
        """Generate class images if prior preservation is enabled"""
        if not self.args.with_prior_preservation:
            return

        class_images_dir = Path(self.args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < self.args.num_class_images:
            torch_dtype = (
                torch.float16
                if self.accelerator.device.type in ("cuda", "xpu")
                else torch.float32
            )
            if self.args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif self.args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif self.args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(
                self.args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=self.args.revision,
                variant=self.args.variant,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = self.args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(self.args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset, batch_size=self.args.sample_batch_size
            )

            sample_dataloader = self.accelerator.prepare(sample_dataloader)
            pipeline.to(self.accelerator.device)

            for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not self.accelerator.is_local_main_process,
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = (
                        class_images_dir
                        / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    )
                    image.save(image_filename)

            del pipeline
            free_memory()

    def setup_models_and_tokenizer(self):
        """Load and setup models, tokenizer"""
        # Load the tokenizer
        if self.args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(
                self.args.tokenizer_name, revision=self.args.revision, use_fast=False
            )
        elif self.args.pretrained_model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                self.args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=self.args.revision,
                use_fast=False,
            )

        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(
            self.args.pretrained_model_name_or_path, self.args.revision
        )

        # Load scheduler and models
        noise_scheduler = DDPMScheduler.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        text_encoder = text_encoder_cls.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.args.revision,
            variant=self.args.variant,
        )
        try:
            vae = AutoencoderKL.from_pretrained(
                self.args.pretrained_model_name_or_path,
                subfolder="vae",
                revision=self.args.revision,
                variant=self.args.variant,
            )
        except OSError:
            # IF does not have a VAE so let's just set it to None
            # We don't have to error out here
            vae = None

        unet = UNet2DConditionModel.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.args.revision,
            variant=self.args.variant,
        )

        return tokenizer, text_encoder, vae, unet, noise_scheduler

    def setup_lora(self, unet, text_encoder):
        """Setup LoRA configuration for unet and text encoder"""
        # We only train the additional adapter LoRA layers
        unet.requires_grad_(False)
        text_encoder.requires_grad_(False)

        # now we will add new LoRA weights to the attention layers
        unet_lora_config = LoraConfig(
            r=self.args.rank,
            lora_alpha=self.args.rank,
            lora_dropout=self.args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
        )
        unet.add_adapter(unet_lora_config)

        # The text encoder comes from 🤗 transformers, we will also attach adapters to it.
        if self.args.train_text_encoder:
            text_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.rank,
                lora_dropout=self.args.lora_dropout,
                init_lora_weights="gaussian",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            )
            text_encoder.add_adapter(text_lora_config)

    def create_save_load_hooks(self, unet, text_encoder):
        """Create custom saving & loading hooks for accelerator"""

        def unwrap_model(model):
            model = self.accelerator.unwrap_model(model)
            model = model._orig_mod if is_compiled_module(model) else model
            return model

        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                # there are only two options here. Either are just the unet attn processor layers
                # or there are the unet and text encoder atten layers
                unet_lora_layers_to_save = None
                text_encoder_lora_layers_to_save = None

                for model in models:
                    if isinstance(model, type(unwrap_model(unet))):
                        unet_lora_layers_to_save = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)
                        )
                    elif isinstance(model, type(unwrap_model(text_encoder))):
                        text_encoder_lora_layers_to_save = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)
                        )
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

                StableDiffusionLoraLoaderMixin.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                    text_encoder_lora_layers=text_encoder_lora_layers_to_save,
                )

        def load_model_hook(models, input_dir):
            unet_ = None
            text_encoder_ = None

            while len(models) > 0:
                model = models.pop()

                if isinstance(model, type(unwrap_model(unet))):
                    unet_ = model
                elif isinstance(model, type(unwrap_model(text_encoder))):
                    text_encoder_ = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

            lora_state_dict, network_alphas = StableDiffusionLoraLoaderMixin.lora_state_dict(
                input_dir
            )

            unet_state_dict = {
                f"{k.replace('unet.', '')}": v
                for k, v in lora_state_dict.items()
                if k.startswith("unet.")
            }
            unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
            incompatible_keys = set_peft_model_state_dict(
                unet_, unet_state_dict, adapter_name="default"
            )

            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

            if self.args.train_text_encoder:
                _set_state_dict_into_text_encoder(
                    lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_
                )

            # Make sure the trainable params are in float32. This is again needed since the base models
            # are in `weight_dtype`. More details:
            # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
            if self.args.mixed_precision == "fp16":
                models = [unet_]
                if self.args.train_text_encoder:
                    models.append(text_encoder_)

                # only upcast trainable parameters (LoRA) into fp32
                cast_training_params(models, dtype=torch.float32)

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def train(self):
        """Main training function"""
        # Setup accelerator
        self.setup_accelerator()
        self.validate_args()
        self.setup_logging()

        # If passed along, set the training seed now.
        if self.args.seed is not None:
            set_seed(self.args.seed)

        # Generate class images if prior preservation is enabled
        self.generate_class_images()

        # Handle the repository creation
        if self.accelerator.is_main_process:
            if self.args.output_dir is not None:
                os.makedirs(self.args.output_dir, exist_ok=True)

            if self.args.push_to_hub:
                repo_id = create_repo(
                    repo_id=self.args.hub_model_id or Path(self.args.output_dir).name,
                    exist_ok=True,
                    token=self.args.hub_token,
                ).repo_id

        # Load models and tokenizer
        tokenizer, text_encoder, vae, unet, noise_scheduler = self.setup_models_and_tokenizer()

        # Setup LoRA
        self.setup_lora(unet, text_encoder)

        # Freeze VAE
        if vae is not None:
            vae.requires_grad_(False)

        # For mixed precision training we cast all non-trainable weights to half-precision
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move models to device and cast to weight_dtype
        unet.to(self.accelerator.device, dtype=weight_dtype)
        if vae is not None:
            vae.to(self.accelerator.device, dtype=weight_dtype)
        text_encoder.to(self.accelerator.device, dtype=weight_dtype)

        # Enable xformers memory efficient attention if requested
        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warning(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, "
                        "please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        # Enable gradient checkpointing if requested
        if self.args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            if self.args.train_text_encoder:
                text_encoder.gradient_checkpointing_enable()

        # Setup save/load hooks
        self.create_save_load_hooks(unet, text_encoder)

        # Enable TF32 for faster training on Ampere GPUs
        if self.args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # Scale learning rate if requested
        if self.args.scale_lr:
            self.args.learning_rate = (
                self.args.learning_rate
                * self.args.gradient_accumulation_steps
                * self.args.train_batch_size
                * self.accelerator.num_processes
            )

        # Make sure the trainable params are in float32
        if self.args.mixed_precision == "fp16":
            models = [unet]
            if self.args.train_text_encoder:
                models.append(text_encoder)

            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models, dtype=torch.float32)

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if self.args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # Optimizer creation
        params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
        if self.args.train_text_encoder:
            params_to_optimize = params_to_optimize + list(
                filter(lambda p: p.requires_grad, text_encoder.parameters())
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

        # Handle text embeddings
        if self.args.pre_compute_text_embeddings:

            def compute_text_embeddings(prompt):
                with torch.no_grad():
                    text_inputs = tokenize_prompt(
                        tokenizer, prompt, tokenizer_max_length=self.args.tokenizer_max_length
                    )
                    prompt_embeds = encode_prompt(
                        text_encoder,
                        text_inputs.input_ids,
                        text_inputs.attention_mask,
                        text_encoder_use_attention_mask=self.args.text_encoder_use_attention_mask,
                    )

                return prompt_embeds

            pre_computed_encoder_hidden_states = compute_text_embeddings(self.args.instance_prompt)
            validation_prompt_negative_prompt_embeds = compute_text_embeddings("")

            if self.args.validation_prompt is not None:
                validation_prompt_encoder_hidden_states = compute_text_embeddings(
                    self.args.validation_prompt
                )
            else:
                validation_prompt_encoder_hidden_states = None

            if self.args.class_prompt is not None:
                pre_computed_class_prompt_encoder_hidden_states = compute_text_embeddings(
                    self.args.class_prompt
                )
            else:
                pre_computed_class_prompt_encoder_hidden_states = None

            text_encoder = None
            tokenizer = None

            gc.collect()
            free_memory()
        else:
            pre_computed_encoder_hidden_states = None
            validation_prompt_encoder_hidden_states = None
            validation_prompt_negative_prompt_embeds = None
            pre_computed_class_prompt_encoder_hidden_states = None

        # Dataset and DataLoaders creation
        train_dataset = DreamBoothDataset(
            instance_data_root=self.args.instance_data_dir,
            instance_prompt=self.args.instance_prompt,
            class_data_root=self.args.class_data_dir if self.args.with_prior_preservation else None,
            class_prompt=self.args.class_prompt,
            class_num=self.args.num_class_images,
            tokenizer=tokenizer,
            size=self.args.resolution,
            center_crop=self.args.center_crop,
            encoder_hidden_states=pre_computed_encoder_hidden_states,
            class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
            tokenizer_max_length=self.args.tokenizer_max_length,
            mvtec_name=self.args.mvtec_name,
            mvtec_anamaly_name=self.args.mvtec_anamaly_name,
            image_interpolation_mode=self.args.image_interpolation_mode,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, self.args.with_prior_preservation),
            num_workers=self.args.dataloader_num_workers,
        )

        # Scheduler and math around the number of training steps
        num_warmup_steps_for_scheduler = self.args.lr_warmup_steps * self.accelerator.num_processes
        if self.args.max_train_steps is None:
            len_train_dataloader_after_sharding = math.ceil(
                len(train_dataloader) / self.accelerator.num_processes
            )
            num_update_steps_per_epoch = math.ceil(
                len_train_dataloader_after_sharding / self.args.gradient_accumulation_steps
            )
            num_training_steps_for_scheduler = (
                self.args.num_train_epochs
                * self.accelerator.num_processes
                * num_update_steps_per_epoch
            )
        else:
            num_training_steps_for_scheduler = self.args.max_train_steps * self.accelerator.num_processes

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps_for_scheduler,
            num_training_steps=num_training_steps_for_scheduler,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        # Prepare everything with our `accelerator`.
        if self.args.train_text_encoder:
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.args.gradient_accumulation_steps
        )
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
            if num_training_steps_for_scheduler != self.args.max_train_steps * self.accelerator.num_processes:
                logger.warning(
                    f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                    f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                    f"This inconsistency may result in the learning rate scheduler not functioning properly."
                )

        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            tracker_config = vars(copy.deepcopy(self.args))
            tracker_config.pop("validation_images")
            self.accelerator.init_trackers("dreambooth-lora-stage1", config=tracker_config)

        # Train!
        total_batch_size = (
            self.args.train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if self.args.resume_from_checkpoint:
            if self.args.resume_from_checkpoint != "latest":
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(self.args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.args.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(self.args.output_dir, path))
                global_step = int(path.split("-")[1])

                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch
        else:
            initial_global_step = 0

        progress_bar = tqdm(
            range(0, self.args.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
        )

        for epoch in range(first_epoch, self.args.num_train_epochs):
            unet.train()
            if self.args.train_text_encoder:
                text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(unet):
                    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

                    if vae is not None:
                        # Convert images to latent space
                        model_input = vae.encode(pixel_values).latent_dist.sample()
                        model_input = model_input * vae.config.scaling_factor
                    else:
                        model_input = pixel_values

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    bsz, channels, height, width = model_input.shape
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                    timesteps = timesteps.long()

                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                    # Get the text embedding for conditioning
                    if self.args.pre_compute_text_embeddings:
                        encoder_hidden_states = batch["input_ids"]
                    else:
                        encoder_hidden_states = encode_prompt(
                            text_encoder,
                            batch["input_ids"],
                            batch["attention_mask"],
                            text_encoder_use_attention_mask=self.args.text_encoder_use_attention_mask,
                        )

                    if self.accelerator.unwrap_model(unet).config.in_channels == channels * 2:
                        noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

                    if self.args.class_labels_conditioning == "timesteps":
                        class_labels = timesteps
                    else:
                        class_labels = None

                    # Predict the noise residual
                    model_pred = unet(
                        noisy_model_input,
                        timesteps,
                        encoder_hidden_states,
                        class_labels=class_labels,
                        return_dict=False,
                    )[0]

                    # if model predicts variance, throw away the prediction. we will only train on the
                    # simplified training objective. This means that all schedulers using the fine tuned
                    # model must be configured to use one of the fixed variance variance types.
                    if model_pred.shape[1] == 6:
                        model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    if self.args.with_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        # Compute instance loss
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                        # Compute prior loss
                        prior_loss = F.mse_loss(
                            model_pred_prior.float(), target_prior.float(), reduction="mean"
                        )

                        # Add the prior loss to the instance loss.
                        loss = loss + self.args.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(params_to_optimize, self.args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if self.accelerator.is_main_process:
                        if global_step % self.args.checkpointing_steps == 0:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if self.args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(self.args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= self.args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - self.args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(
                                            self.args.output_dir, removing_checkpoint
                                        )
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
                            self.accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= self.args.max_train_steps:
                    break

            if self.accelerator.is_main_process:
                if (
                    self.args.validation_prompt is not None
                    and epoch % self.args.validation_epochs == 0
                ):
                    # create pipeline
                    pipeline = DiffusionPipeline.from_pretrained(
                        self.args.pretrained_model_name_or_path,
                        unet=self.accelerator.unwrap_model(unet),
                        text_encoder=(
                            None
                            if self.args.pre_compute_text_embeddings
                            else self.accelerator.unwrap_model(text_encoder)
                        ),
                        revision=self.args.revision,
                        variant=self.args.variant,
                        torch_dtype=weight_dtype,
                    )

                    if self.args.pre_compute_text_embeddings:
                        pipeline_args = {
                            "prompt_embeds": validation_prompt_encoder_hidden_states,
                            "negative_prompt_embeds": validation_prompt_negative_prompt_embeds,
                            "num_inference_steps": self.args.num_inference_steps,
                        }
                    else:
                        pipeline_args = {
                            "prompt": self.args.validation_prompt,
                            "num_inference_steps": self.args.num_inference_steps,
                        }

                    images = log_validation(
                        pipeline,
                        self.args,
                        self.accelerator,
                        pipeline_args,
                        epoch,
                        torch_dtype=weight_dtype,
                    )

        # Save the lora layers
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unet = self.accelerator.unwrap_model(unet)
            unet = unet.to(torch.float32)

            unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

            if self.args.train_text_encoder:
                text_encoder = self.accelerator.unwrap_model(text_encoder)
                text_encoder_state_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(text_encoder)
                )
            else:
                text_encoder_state_dict = None

            StableDiffusionLoraLoaderMixin.save_lora_weights(
                save_directory=self.args.output_dir,
                unet_lora_layers=unet_lora_state_dict,
                text_encoder_lora_layers=text_encoder_state_dict,
            )

            # Final inference
            # Load previous pipeline
            pipeline = DiffusionPipeline.from_pretrained(
                self.args.pretrained_model_name_or_path,
                revision=self.args.revision,
                variant=self.args.variant,
                torch_dtype=weight_dtype,
            )

            # load attention processors
            pipeline.load_lora_weights(self.args.output_dir, weight_name="pytorch_lora_weights.safetensors")

            # run inference
            images = []
            if self.args.validation_prompt and self.args.num_validation_images > 0:
                pipeline_args = {
                    "prompt": self.args.validation_prompt,
                    "num_inference_steps": self.args.num_inference_steps,
                }
                images = log_validation(
                    pipeline,
                    self.args,
                    self.accelerator,
                    pipeline_args,
                    self.args.num_train_epochs,
                    is_final_validation=True,
                    torch_dtype=weight_dtype,
                )

            if self.args.push_to_hub:
                save_model_card(
                    repo_id,
                    images=images,
                    base_model=self.args.pretrained_model_name_or_path,
                    train_text_encoder=self.args.train_text_encoder,
                    prompt=self.args.instance_prompt,
                    repo_folder=self.args.output_dir,
                    pipeline=pipeline,
                )
                upload_folder(
                    repo_id=repo_id,
                    folder_path=self.args.output_dir,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                )

        self.accelerator.end_training()