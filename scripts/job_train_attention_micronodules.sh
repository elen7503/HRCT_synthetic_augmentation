#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=12:00:00
#PBS -N train_attention_micronodules
#PBS -o $PBS_O_WORKDIR/logs/train_attention_micronodules.out
#PBS -e $PBS_O_WORKDIR/logs/train_attention_micronodules.err
: "${PROJECT_ROOT:?PROJECT_ROOT no está definida — exporta esta variable antes de lanzar el job}"
module load CUDA/12.1.1
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild
cd ${PROJECT_ROOT}/diffusion_model
echo "Training micronodules (attention U-Net) | Start: $(date)"
python train.py --class_idx 4 \
    --epochs 1000 \
    --save_every 50 \
    --output_dir outputs_attention
echo "End: $(date)"
