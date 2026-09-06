#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=04:00:00
#PBS -N ema_finetune_emphysema
#PBS -o $PBS_O_WORKDIR/logs/ema_finetune_emphysema.out
#PBS -e $PBS_O_WORKDIR/logs/ema_finetune_emphysema.err
: "${PROJECT_ROOT:?PROJECT_ROOT no está definida — exporta esta variable antes de lanzar el job}"
module load CUDA/12.1.1
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild
cd ${PROJECT_ROOT}/diffusion_model
python train_ema_finetune.py --class_idx 1 \
    --weightfile outputs/emphysema/best.pt \
    --output_dir outputs/emphysema_ema \
    --extra_epochs 300
