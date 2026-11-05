# Modified from infer.py of RealFill (https://github.com/thuanz123/realfill),
# Copyright (c) 2023 Thuan Hoang Nguyen, released under the CreativeML Open RAIL++-M License (see LICENSE).
# Changes: two-pass DDIM inference and image metrics.

import argparse
import csv
import json
import os

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from skimage.metrics import structural_similarity
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils.torch_utils import randn_tensor
from safetensors.torch import load_file
from transformers import AutoTokenizer, CLIPTextModel

from train_UnetTwice import TRAINED_UNET_PARTS, build_ref_attention, unet_twice

parser = argparse.ArgumentParser(description="Inference")
parser.add_argument(
    "--pretrained_model_name_or_path",
    type=str,
    default=None,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    required=True,
    help="The training output directory, containing `folder_<i>/checkpoint-<step>/`",
)
parser.add_argument(
    "--train_data_dir",
    type=str,
    default=None,
    required=True,
    help="The directory containing the numbered data folders",
)
parser.add_argument("--folders", type=str, default="0-9", help="Range of data folders to evaluate, e.g. `0-9`")
parser.add_argument(
    "--checkpoints", type=str, default="2000", help="Comma-separated list of checkpoint steps to evaluate"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./test-infer/",
    help="The output directory where predictions and metrics are saved",
)
parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible inference.")
parser.add_argument("--num_inference_steps", type=int, default=50)
parser.add_argument("--eta", type=float, default=1.0, help="The eta of the DDIM scheduler.")
parser.add_argument(
    "--num_ref_skips",
    type=int,
    default=0,
    help="The `--num_ref_skips` the models were trained with: skip-connection features passed from the reference pass.",
)
parser.add_argument(
    "--num_views",
    type=int,
    default=1,
    help="The `--num_views` the models were trained with: the first `num_views` images of `refs` in the first line of"
    " `pairs.jsonl` are used (`ref` alone for 1).",
)
parser.add_argument(
    "--ref_feature_scale",
    type=float,
    default=1.0,
    help="What the reference features are multiplied by before being added, normally the `--ref_feature_scale` of"
    " training.",
)
parser.add_argument("--prompt", type=str, default="a photo of sks")
parser.add_argument(
    "--blur_mask",
    action="store_true",
    help="Grow the mask by one pixel (MaxFilter 3) and blur it (BoxBlur 1), then blend the result with the soft mask."
    " By default the mask is used as is and the result is blended with the hard mask.",
)

args = parser.parse_args()

def load_image(path, mode="RGB"):
    return np.array(Image.open(path).convert(mode))

def save_image(image, path):
    image = (image.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(image).save(path)

@torch.no_grad()
def inpaint(
    unet, vae, scheduler, encoder_hidden_states, image, mask, ref_images, num_inference_steps, eta, num_ref_skips,
    ref_attention=None, ref_scale=1.0, generator=None,
):
    """Denoise with reference features and return an image in [0, 1]."""
    masked_image = image * (mask < 0.5)
    conditionings = vae.encode(masked_image).latent_dist.sample(generator) * vae.config.scaling_factor
    ref_conditionings = [
        vae.encode(ref_image * (mask > 0.5)).latent_dist.sample(generator) * vae.config.scaling_factor
        for ref_image in ref_images
    ]
    masks = F.interpolate(mask, size=conditionings.shape[2:])

    latents = randn_tensor(conditionings.shape, generator=generator, device=conditionings.device)
    scheduler.set_timesteps(num_inference_steps, device=latents.device)
    for t in scheduler.timesteps:
        inputs = torch.cat([latents, masks, conditionings], dim=1)
        ref_inputs = [torch.cat([latents, torch.zeros_like(masks), ref], dim=1) for ref in ref_conditionings]
        noise_pred = unet_twice(
            unet, inputs, ref_inputs, t, encoder_hidden_states, num_ref_skips, ref_attention, ref_scale
        )
        latents = scheduler.step(noise_pred, t, latents, eta=eta, generator=generator).prev_sample

    image = vae.decode(latents / vae.config.scaling_factor).sample
    return ((image + 1) / 2).clamp(0, 1)

def load_unet(checkpoint_dir):
    """Load the base UNet and trained weights."""
    trained_path = os.path.join(checkpoint_dir, "unet_trained.safetensors")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    missing, unexpected = unet.load_state_dict(load_file(trained_path), strict=False)
    if unexpected or any(k.split(".")[0] in TRAINED_UNET_PARTS for k in missing):
        raise ValueError(f"{trained_path} does not match the fine-tuned parts {TRAINED_UNET_PARTS} of the UNet")
    return unet

def crop_to_mask(image, mask):
    ys, xs = torch.nonzero(mask[0], as_tuple=True)
    return image[:, ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def compute_metrics(result, gt, loss_fn):
    psnr = 10 * torch.log10(1 / torch.mean((result - gt) ** 2))
    ssim = structural_similarity(
        TF.rgb_to_grayscale(result)[0].numpy(), TF.rgb_to_grayscale(gt)[0].numpy(), data_range=1.0
    )
    lpips_value = loss_fn(result[None].cuda() * 2 - 1, gt[None].cuda() * 2 - 1)
    return float(psnr), float(ssim), float(lpips_value)

if __name__ == "__main__":
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False
    )
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(device)
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    loss_fn = lpips.LPIPS(net="vgg").to(device)

    prompt_ids = tokenizer(
        args.prompt, truncation=True, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    ).input_ids
    with torch.no_grad():
        encoder_hidden_states = text_encoder(prompt_ids.to(device))[0]

    first, last = (int(i) for i in args.folders.split("-"))
    folders = list(range(first, last + 1))
    checkpoints = [int(step) for step in args.checkpoints.split(",")]

    rows = []
    for checkpoint in checkpoints:
        for folder in folders:
            checkpoint_dir = os.path.join(args.model_path, f"folder_{folder}", f"checkpoint-{checkpoint}")
            unet = load_unet(checkpoint_dir).to(device)
            ref_attention = None
            if args.num_views > 1:
                ref_attention = build_ref_attention(unet, args.num_views)
                ref_attention.load_state_dict(load_file(os.path.join(checkpoint_dir, "ref_attention.safetensors")))
                ref_attention.to(device).eval()

            data_dir = os.path.join(args.train_data_dir, str(folder))
            with open(os.path.join(data_dir, "pairs.jsonl"), "rt") as f:
                pair = json.loads(f.readline())

            image = load_image(os.path.join(data_dir, pair["target"]))
            gt = load_image(os.path.join(data_dir, "target_gt.png"))
            ref_names = pair["refs"][: args.num_views] if args.num_views > 1 else [pair["ref"]]
            ref_images = [load_image(os.path.join(data_dir, name)) for name in ref_names]
            mask_image = Image.open(os.path.join(data_dir, pair["hole"])).convert("L")

            mask = torch.from_numpy(np.array(mask_image))[None].float() / 255.0

            if args.blur_mask:
                mask_image = mask_image.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.BoxBlur(1))
            blend_mask = torch.from_numpy(np.array(mask_image))[None].float() / 255.0
            fill_mask = (blend_mask >= 0.5).float()

            image_t = torch.from_numpy(image).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1.0
            ref_images_t = [
                torch.from_numpy(ref).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1.0 for ref in ref_images
            ]

            generator = torch.Generator(device=device).manual_seed(args.seed)
            result = inpaint(
                unet, vae, scheduler, encoder_hidden_states, image_t, fill_mask[None].to(device), ref_images_t,
                args.num_inference_steps, args.eta, args.num_ref_skips, ref_attention, args.ref_feature_scale,
                generator,
            )
            result = result[0].float().cpu()

            gt = torch.from_numpy(gt).permute(2, 0, 1).float() / 255.0
            composite = result * blend_mask + torch.from_numpy(image).permute(2, 0, 1).float() / 255.0 * (1 - blend_mask)
            result_hole, gt_hole = crop_to_mask(result, mask), crop_to_mask(gt, mask)

            row = {"folder": folder, "checkpoint": checkpoint}
            row["psnr_generated"], row["ssim_generated"], row["lpips_generated"] = compute_metrics(result, gt, loss_fn)
            row["psnr_composite"], row["ssim_composite"], row["lpips_composite"] = compute_metrics(composite, gt, loss_fn)
            row["psnr_hole"], row["ssim_hole"], row["lpips_hole"] = compute_metrics(result_hole, gt_hole, loss_fn)
            row["hole_area"] = int(torch.count_nonzero(mask))
            rows.append(row)

            save_dir = os.path.join(args.output_dir, f"checkpoint-{checkpoint}", f"folder_{folder}")
            os.makedirs(save_dir, exist_ok=True)
            save_image(result, os.path.join(save_dir, "generated.png"))
            save_image(gt, os.path.join(save_dir, "ground_truth.png"))
            save_image(composite, os.path.join(save_dir, "composite.png"))
            save_image(result_hole, os.path.join(save_dir, "generated_hole.png"))
            save_image(gt_hole, os.path.join(save_dir, "ground_truth_hole.png"))

            del unet
            torch.cuda.empty_cache()

    with open(os.path.join(args.output_dir, "all_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metric_names = [k for k in rows[0] if k not in ("folder", "checkpoint")]
    with open(os.path.join(args.output_dir, "metrics_mean_per_checkpoint.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint"] + [f"{k}_mean" for k in metric_names])
        writer.writeheader()
        for checkpoint in checkpoints:
            values = [row for row in rows if row["checkpoint"] == checkpoint]
            mean_row = {"checkpoint": checkpoint}
            mean_row.update({f"{k}_mean": sum(row[k] for row in values) / len(values) for k in metric_names})
            writer.writerow(mean_row)
