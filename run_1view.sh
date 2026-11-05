#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0

for i in {0..9}; do
  accelerate launch --num_processes=1 --num_machines=1 --mixed_precision=fp16 \
    train_UnetTwice.py \
    --pretrained_model_name_or_path=models/stable-diffusion-2-inpainting \
    --train_data_dir=data/10_aligned_samples/$i \
    --output_dir=outputs/1view/folder_$i
done

python infer.py \
  --pretrained_model_name_or_path=models/stable-diffusion-2-inpainting \
  --model_path=outputs/1view \
  --train_data_dir=data/10_aligned_samples \
  --folders=0-9 \
  --checkpoints=200,500,1000,1500,2000 \
  --output_dir=outputs/1view/results
