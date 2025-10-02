import torch


def collate_fn(examples, with_prior_preservation=False):
    has_attention_mask = "instance_attention_mask_blends" in examples[0]

    input_id_blends = [example["instance_prompt_id_blends"] for example in examples]
    pixel_value_blends = [example["instance_image_blends"] for example in examples]
    input_id_fgs = [example["instance_prompt_id_fgs"] for example in examples]
    pixel_value_fgs = [example["instance_image_fgs"] for example in examples]

    if has_attention_mask:
        attention_mask_blend = [
            example["instance_attention_mask_blends"] for example in examples
        ]
        attention_mask_fg = [
            example["instance_attention_mask_fgs"] for example in examples
        ]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        if has_attention_mask:
            attention_mask += [example["class_attention_mask"] for example in examples]

    pixel_value_blends = torch.stack(pixel_value_blends)
    pixel_value_blends = pixel_value_blends.to(
        memory_format=torch.contiguous_format
    ).float()
    pixel_value_fgs = torch.stack(pixel_value_fgs)
    pixel_value_fgs = pixel_value_fgs.to(memory_format=torch.contiguous_format).float()

    input_id_blends = torch.cat(input_id_blends, dim=0)
    input_id_fgs = torch.cat(input_id_fgs, dim=0)

    batch = {
        "input_id_blends": input_id_blends,
        "pixel_value_blends": pixel_value_blends,
        "input_id_fgs": input_id_fgs,
        "pixel_value_fgs": pixel_value_fgs,
    }

    if has_attention_mask:
        attention_mask_blend = torch.cat(attention_mask_blend, dim=0)
        attention_mask_fg = torch.cat(attention_mask_fg, dim=0)
        batch["attention_mask_blend"] = attention_mask_blend
        batch["attention_mask_fg"] = attention_mask_fg

    return batch