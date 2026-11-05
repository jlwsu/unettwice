# Modified from train_realfill.py of RealFill (https://github.com/thuanz123/realfill),
# Copyright (c) 2023 Thuan Hoang Nguyen, released under the CreativeML Open RAIL++-M License (see LICENSE).
# Changes: shared two-pass UNet, decoder tuning, and multi-view fusion.

import random
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms_v2
from transformers import AutoTokenizer, CLIPTextModel

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from safetensors.torch import save_file

check_min_version("0.20.1")

TRAINED_UNET_PARTS = ("up_blocks", "conv_norm_out", "conv_out")

def make_mask(images, resolution, times=30):
    mask, times = torch.ones_like(images[0:1, :, :]), np.random.randint(1, times)
    min_size, max_size, margin = np.array([0.03, 0.25, 0.01]) * resolution
    max_size = min(max_size, resolution - margin * 2)

    for _ in range(times):
        width = np.random.randint(int(min_size), int(max_size))
        height = np.random.randint(int(min_size), int(max_size))

        x_start = np.random.randint(int(margin), resolution - int(margin) - width + 1)
        y_start = np.random.randint(int(margin), resolution - int(margin) - height + 1)
        mask[:, y_start:y_start + height, x_start:x_start + width] = 0

    mask = 1 - mask if random.random() < 0.5 else mask
    return mask

def get_ref_features(unet, sample, timesteps, encoder_hidden_states):
    """Return reference skip and mid-block features."""
    timesteps = timesteps.expand(sample.shape[0])
    emb = unet.time_embedding(unet.time_proj(timesteps).to(dtype=sample.dtype))

    sample = unet.conv_in(sample)
    skip_features = (sample,)
    for downsample_block in unet.down_blocks:
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            sample, res_samples = downsample_block(
                hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
        skip_features += res_samples

    return skip_features, unet.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

class RefSelfAttention(nn.Module):
    """Spatial self-attention with GroupNorm, 1x1 QKV, and a residual projection."""

    def __init__(self, channels):
        super().__init__()
        num_groups = 32
        while channels % num_groups != 0:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        hidden = self.norm(x)
        q, k, v = self.q(hidden), self.k(hidden), self.v(hidden)
        batch, channels, height, width = q.shape
        q = q.flatten(2).transpose(1, 2)
        weights = torch.bmm(q, k.flatten(2)) * channels ** (-0.5)
        weights = torch.softmax(weights, dim=2)
        hidden = torch.bmm(v.flatten(2), weights.transpose(1, 2))
        return x + self.proj_out(hidden.reshape(batch, channels, height, width))

def build_ref_attention(unet, num_views):
    """Build attention modules for each feature width."""
    widths = dict.fromkeys(unet.config.block_out_channels)
    return nn.ModuleDict({str(num_views * width): RefSelfAttention(num_views * width) for width in widths})

def fuse_ref_features(ref_attention, features):
    """Fuse views and return the first view channels."""
    channels = features[0].shape[1]
    attention = ref_attention[str(len(features) * channels)]
    fused = attention(torch.cat(features, dim=1).to(next(attention.parameters()).dtype))
    return fused[:, :channels].to(features[0].dtype)

def unet_twice(
    unet, inputs, ref_inputs, timesteps, encoder_hidden_states, num_ref_skips=0, ref_attention=None, ref_scale=1.0
):
    """Inject reference features into a second pass of the same UNet."""
    if torch.is_tensor(ref_inputs):
        ref_inputs = [ref_inputs]

    with torch.no_grad():
        ref_features = [get_ref_features(unet, x, timesteps, encoder_hidden_states) for x in ref_inputs]
    skip_features, mid_feature = ref_features[0]

    if not 0 <= num_ref_skips <= len(skip_features):
        raise ValueError(f"num_ref_skips must be between 0 and {len(skip_features)}, got {num_ref_skips}")
    num_unused = len(skip_features) - num_ref_skips

    if len(ref_inputs) > 1:
        if ref_attention is None:
            raise ValueError("Several reference views need `ref_attention`")
        view_skips = [skips for skips, _ in ref_features]
        skip_features = list(skip_features[:num_unused]) + [
            fuse_ref_features(ref_attention, [skips[i] for skips in view_skips])
            for i in range(num_unused, len(skip_features))
        ]
        mid_feature = fuse_ref_features(ref_attention, [mid for _, mid in ref_features])

    if ref_scale != 1:
        skip_features = [f * ref_scale for f in skip_features]
        mid_feature = mid_feature * ref_scale

    skip_residuals = [torch.zeros_like(f) for f in skip_features[:num_unused]] + list(skip_features[num_unused:])
    return unet(
        inputs,
        timesteps,
        encoder_hidden_states,
        down_block_additional_residuals=skip_residuals,
        mid_block_additional_residual=mid_feature,
    ).sample

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Fine-tune the two-pass inpainting UNet on one scene.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="unet-twice-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--max_train_steps", type=int, default=2000, help="Total number of training steps to perform."
    )
    parser.add_argument(
        "--ref_feature_scale",
        type=float,
        default=1.0,
        help="How strongly the reference features of the first pass are added to the second pass: they are multiplied"
        " by this value before being added. 1 (default) adds them as they are.",
    )
    parser.add_argument(
        "--empty_prompt_prob",
        type=float,
        default=0.1,
        help="The probability of training an example with an empty prompt instead of its prompt (default 0.1; 0 turns it off).",
    )
    parser.add_argument(
        "--num_ref_skips",
        type=int,
        default=0,
        help=(
            "How many skip-connection features of the reference pass, counted from the mid block (0-12), are added"
            " to the second pass besides the mid-block output. 0 passes the mid-block output only."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="200,500,1000,1500,2000",
        help="Comma-separated list of training steps at which the UNet is saved, e.g. `200,500,1000`.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--unet_learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate to use for unet.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=None, type=float, help="Max gradient norm. No clipping if not set.")
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision: fp16 (default), bf16 (needs an Nvidia Ampere GPU) or no (full fp32 training).",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument(
        "--fast_mode",
        type=int,
        default=0,
        choices=[0, 2, 4, 8],
        help=(
            "0 (default) builds and encodes every example of the batch. 2, 4 or 8 sets the batch size to this value"
            " and the training steps to 200: each step builds the two training pairs once, encodes them with the VAE"
            " once and repeats them to fill the batch. The noise and the timesteps are still drawn per example."
        ),
    )

    parser.add_argument(
        "--num_views",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help=(
            "How many reference views the first pass reads. 1 (default) uses the `ref` of the first two lines of"
            " `pairs.jsonl`; more uses all its lines with the first `num_views` images of their `refs`, fused by"
            " self-attention."
        ),
    )

    args = parser.parse_args(input_args)

    if args.fast_mode and args.num_views > 1:
        raise ValueError("--fast_mode only supports --num_views 1")

    if args.fast_mode:
        args.train_batch_size = args.fast_mode
        args.max_train_steps = 200

    if max(int(step) for step in args.checkpointing_steps.split(",")) > args.max_train_steps:
        raise ValueError(
            f"--checkpointing_steps {args.checkpointing_steps} goes beyond the {args.max_train_steps} training steps"
        )

    return args

class UnetTwiceDataset(Dataset):
    """Load one scene from pairs.jsonl."""

    def __init__(
        self,
        train_data_root,
        tokenizer,
        size=512,
        num_examples=8,
        num_views=1,
        empty_prompt_prob=0.1,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.empty_prompt_prob = empty_prompt_prob

        self.train_data_root = Path(train_data_root)
        if not (self.train_data_root / "pairs.jsonl").exists():
            raise ValueError(f"{self.train_data_root / 'pairs.jsonl'} doesn't exist.")

        with open(self.train_data_root / "pairs.jsonl", "rt") as f:
            self.train_pairs = [json.loads(line) for line in f if line.strip()]
        if num_views == 1:
            self.train_pairs = self.train_pairs[:2]
            self.ref_names = [[pair["ref"]] for pair in self.train_pairs]
        else:
            self.ref_names = [pair["refs"][:num_views] for pair in self.train_pairs]
            for pair, names in zip(self.train_pairs, self.ref_names):
                if len(names) != num_views:
                    raise ValueError(f"`refs` must have at least {num_views} images: {pair}")

        self.num_examples = math.ceil(num_examples / len(self.train_pairs)) * len(self.train_pairs)

        self.images = {}
        for pair, names in zip(self.train_pairs, self.ref_names):
            files = [(pair["target"], "RGB"), (pair["hole"], "L"), (pair["loss_weight"], "L")]
            for key in files + [(name, "RGB") for name in names]:
                if key not in self.images:
                    self.images[key] = self.load_image(*key)

        self.transform = transforms_v2.Compose(
            [
                transforms_v2.RandomResize(size, int(1.125 * size)),
                transforms_v2.RandomCrop(size),
                transforms_v2.ToImageTensor(),
                transforms_v2.ConvertImageDtype(),
                transforms_v2.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self.num_examples

    def load_image(self, file_name, mode):
        image = Image.open(self.train_data_root / file_name)
        image = exif_transpose(image)
        return image.convert(mode)

    def get_image(self, file_name, mode):
        """Copy a cached image."""
        return self.images[(file_name, mode)].copy()

    def __getitem__(self, index):
        example = {}
        line = index % len(self.train_pairs)
        pair = self.train_pairs[line]

        image = self.get_image(pair["target"], "RGB")
        ref_images = [self.get_image(name, "RGB") for name in self.ref_names[line]]
        hole = self.get_image(pair["hole"], "L")
        weighting = self.get_image(pair["loss_weight"], "L")

        image, *ref_images, hole, weighting = self.transform(image, *ref_images, hole, weighting)
        hole = (hole[0:1] > 0).float()
        example["images"], example["weightings"] = image, weighting[0:1] > 0

        mask = make_mask(example["images"], self.size)

        if line % 2 == 0:
            example["masks"] = (1 - hole) * mask
            example["conditioning_images"] = example["images"] * (1 - hole) * (mask < 0.5)
        else:
            example["masks"] = mask
            example["conditioning_images"] = example["images"] * (mask < 0.5)

        example["ref_images"] = torch.stack([ref_image * (mask > 0.5) for ref_image in ref_images])

        train_prompt = pair["prompt"]
        if self.empty_prompt_prob > 0 and random.random() < self.empty_prompt_prob:
            train_prompt = ""
        example["prompt_ids"] = self.tokenizer(
            train_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return example

def collate_fn(examples):
    input_ids = [example["prompt_ids"] for example in examples]
    images = [example["images"] for example in examples]

    masks = [example["masks"] for example in examples]
    weightings = [example["weightings"] for example in examples]
    conditioning_images = [example["conditioning_images"] for example in examples]
    ref_images = [example["ref_images"] for example in examples]

    images = torch.stack(images)
    images = images.to(memory_format=torch.contiguous_format).float()

    masks = torch.stack(masks)
    masks = masks.to(memory_format=torch.contiguous_format).float()

    weightings = torch.stack(weightings)
    weightings = weightings.to(memory_format=torch.contiguous_format).float()

    conditioning_images = torch.stack(conditioning_images)
    conditioning_images = conditioning_images.to(memory_format=torch.contiguous_format).float()

    ref_images = torch.stack(ref_images)
    ref_images = ref_images.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "images": images,
        "masks": masks,
        "weightings": weightings,
        "conditioning_images": conditioning_images,
        "ref_images": ref_images,
    }
    return batch

def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    unet.requires_grad_(False)
    for name in TRAINED_UNET_PARTS:
        getattr(unet, name).requires_grad_(True)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if not is_xformers_available():
            raise ValueError("xformers is not available. Make sure it is installed correctly")
        unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    ref_attention = build_ref_attention(unet, args.num_views) if args.num_views > 1 else None

    params_to_optimize = [p for p in unet.parameters() if p.requires_grad]
    param_groups = [{"params": params_to_optimize, "lr": args.unet_learning_rate}]
    if ref_attention is not None:
        param_groups.append({"params": list(ref_attention.parameters()), "lr": args.unet_learning_rate})
        params_to_optimize = params_to_optimize + list(ref_attention.parameters())
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    examples_per_step = 2 if args.fast_mode else args.train_batch_size
    repeats = args.train_batch_size // examples_per_step
    train_dataset = UnetTwiceDataset(
        train_data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        num_examples=examples_per_step,
        num_views=args.num_views,
        empty_prompt_prob=args.empty_prompt_prob,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=examples_per_step,
        shuffle=True,
        collate_fn=collate_fn,

        num_workers=0,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )

    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    if ref_attention is not None:
        ref_attention.to(accelerator.device)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    checkpointing_steps = [int(step) for step in args.checkpointing_steps.split(",")]

    for epoch in range(num_train_epochs):
        unet.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                with accelerator.autocast():

                    latents = vae.encode(batch["images"]).latent_dist.sample()
                    latents = latents * 0.18215

                    conditionings = vae.encode(batch["conditioning_images"]).latent_dist.sample()
                    conditionings = conditionings * 0.18215

                    ref_images = batch["ref_images"]
                    ref_conditionings = torch.stack(
                        [vae.encode(ref_images[:, view]).latent_dist.sample() for view in range(ref_images.shape[1])],
                        dim=1,
                    )
                    ref_conditionings = ref_conditionings * 0.18215

                    masks, size = batch["masks"], latents.shape[2:]
                    masks = F.interpolate(masks, size=size)

                    weightings = batch["weightings"]
                    weightings = F.interpolate(weightings, size=size)

                    if repeats > 1:
                        latents, conditionings, ref_conditionings, masks, weightings = (
                            x.repeat(repeats, *[1] * (x.dim() - 1))
                            for x in (latents, conditionings, ref_conditionings, masks, weightings)
                        )

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]

                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                    )
                    timesteps = timesteps.long()

                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    inputs = torch.cat([noisy_latents, masks, conditionings], dim=1)
                    ref_inputs = [
                        torch.cat([noisy_latents, torch.zeros_like(masks), ref_conditionings[:, view]], dim=1)
                        for view in range(ref_conditionings.shape[1])
                    ]

                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                    if repeats > 1:
                        encoder_hidden_states = encoder_hidden_states.repeat(repeats, 1, 1)

                    model_pred = unet_twice(
                        unet,
                        inputs,
                        ref_inputs,
                        timesteps,
                        encoder_hidden_states,
                        args.num_ref_skips,
                        ref_attention,
                        args.ref_feature_scale,
                    )

                    assert noise_scheduler.config.prediction_type == "epsilon"
                    loss = weightings * F.mse_loss(model_pred.float(), noise.float(), reduction="none")
                    loss = loss.sum(dim=[1, 2, 3]) / (weightings.sum(dim=[1, 2, 3]).clamp(min=1) * loss.shape[1])
                    loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1

                if accelerator.is_main_process:
                    if global_step in checkpointing_steps:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        unet_state = accelerator.unwrap_model(unet).state_dict()
                        trained_state = {k: v for k, v in unet_state.items() if k.split(".")[0] in TRAINED_UNET_PARTS}
                        save_file(trained_state, os.path.join(save_path, "unet_trained.safetensors"))
                        if ref_attention is not None:
                            save_file(ref_attention.state_dict(), os.path.join(save_path, "ref_attention.safetensors"))

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()

if __name__ == "__main__":
    args = parse_args()
    main(args)
