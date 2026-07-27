#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=04:00:00
#PBS -N ema_finetune_healthy
#PBS -o /rds/general/user/eh1121/home/Final_Project/logs/ema_finetune_healthy.out
#PBS -e /rds/general/user/eh1121/home/Final_Project/logs/ema_finetune_healthy.err
module load CUDA/12.1.1
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild
cd /rds/general/user/eh1121/home/Final_Project/diffusion_model
python train_ema_finetune.py --class_idx 0 \
    --weightfile outputs/healthy/best.pt \
    --output_dir outputs/healthy_ema \
    --extra_epochs 300
