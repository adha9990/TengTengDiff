import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision import transforms
from ..utils.tokenization import tokenize_prompt


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt_blend,
        instance_prompt_fg,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        instance_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
        mvtec_name=None,
        mvtec_anamaly_name=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.instance_prompt_encoder_hidden_states = (
            instance_prompt_encoder_hidden_states
        )
        self.tokenizer_max_length = tokenizer_max_length

        self.instance_data_root = instance_data_root
        self.mvtec_name = mvtec_name
        self.mvtec_anamaly_name = mvtec_anamaly_name

        self.instance_images_path = []
        for type_name in os.listdir(self.instance_data_root):
            if type_name not in [mvtec_name]:
                continue
            if not os.path.isdir(
                os.path.join(self.instance_data_root, type_name, "test")
            ):
                continue
            for anamaly_name in os.listdir(
                os.path.join(self.instance_data_root, type_name, "test")
            ):
                if anamaly_name != "good" and anamaly_name == mvtec_anamaly_name:
                    file_names = os.listdir(
                        os.path.join(
                            self.instance_data_root, type_name, "test", anamaly_name
                        )
                    )
                    file_names.sort()
                    n = len(file_names)
                    n = n // 3
                    if (
                        n % 3
                    ):  # 并不是每个类别都需要，这里需要结合所有异常类别的综合数量来指定数据
                        n += 1
                    for name in file_names[:n]:
                        self.instance_images_path.append(
                            os.path.join(
                                self.instance_data_root,
                                type_name,
                                "test",
                                anamaly_name,
                                name,
                            )
                        )
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt_blend = instance_prompt_blend
        self.instance_prompt_fg = instance_prompt_fg
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        if type_name == "cable":
            self.image_transforms = transforms.Compose(
                [
                    transforms.Resize(
                        size, interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    (
                        transforms.CenterCrop(size)
                        if center_crop
                        else transforms.RandomCrop(size)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            self.image_transforms_mask = transforms.Compose(
                [
                    transforms.Resize(
                        size, interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    (
                        transforms.CenterCrop(size)
                        if center_crop
                        else transforms.RandomCrop(size)
                    ),
                    transforms.ToTensor(),
                ]
            )

            self.image_transforms_gt = transforms.Compose(
                [
                    transforms.Resize(
                        64, interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    (
                        transforms.CenterCrop(64)
                        if center_crop
                        else transforms.RandomCrop(64)
                    ),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.image_transforms = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.Resize(
                        size, interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    (
                        transforms.CenterCrop(size)
                        if center_crop
                        else transforms.RandomCrop(size)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            self.image_transforms_mask = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.Resize(
                        size, interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    (
                        transforms.CenterCrop(size)
                        if center_crop
                        else transforms.RandomCrop(size)
                    ),
                    transforms.ToTensor(),
                ]
            )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image_blend = Image.open(
            self.instance_images_path[index % self.num_instance_images]
        )
        instance_image_blend = exif_transpose(instance_image_blend)
        mask = Image.open(
            self.instance_images_path[index % self.num_instance_images]
            .replace("test", "ground_truth")
            .replace(".png", "_mask.png")
        )

        if not instance_image_blend.mode == "RGB":
            instance_image_blend = instance_image_blend.convert("RGB")
        if not mask.mode == "L":
            mask = mask.convert("L")

        # transform imgs
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        example["instance_image_blends"] = self.image_transforms(instance_image_blend)
        torch.random.manual_seed(seed)
        mask = self.image_transforms_mask(mask)
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        example["instance_image_fgs"] = example["instance_image_blends"] * mask

        text_input_blends = tokenize_prompt(
            self.tokenizer,
            self.instance_prompt_blend,
            tokenizer_max_length=self.tokenizer_max_length,
        )
        example["instance_prompt_id_blends"] = text_input_blends.input_ids
        example["instance_attention_mask_blends"] = text_input_blends.attention_mask

        text_input_fgs = tokenize_prompt(
            self.tokenizer,
            self.instance_prompt_fg,
            tokenizer_max_length=self.tokenizer_max_length,
        )
        example["instance_prompt_id_fgs"] = text_input_fgs.input_ids
        example["instance_attention_mask_fgs"] = text_input_fgs.attention_mask

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            if self.instance_prompt_encoder_hidden_states is not None:
                example["class_prompt_ids"] = self.instance_prompt_encoder_hidden_states
            else:
                class_text_inputs = tokenize_prompt(
                    self.tokenizer,
                    self.class_prompt,
                    tokenizer_max_length=self.tokenizer_max_length,
                )
                example["class_prompt_ids"] = class_text_inputs.input_ids
                example["class_attention_mask"] = class_text_inputs.attention_mask

        return example


class PromptDataset(Dataset):
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example
