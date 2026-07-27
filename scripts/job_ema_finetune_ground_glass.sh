#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=04:00:00
#PBS -N ema_finetune_ground_glass
#PBS -o /rds/general/user/eh1121/home/Final_Project/logs/ema_finetune_ground_glass.out
#PBS -e /rds/general/user/eh1121/home/Final_Project/logs/ema_finetune_ground_glass.err
module load CUDA/12.1.1
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild
cd /rds/general/user/eh1121/home/Final_Project/diffusion_model
python train_ema_finetune.py --class_idx 2 \
    --weightfile outputs/ground_glass/best.pt \
    --output_dir outputs/ground_glass_ema \
    --extra_epochs 300
