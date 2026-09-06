#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=12:00:00
#PBS -N train_attention_fibrosis
#PBS -o $PBS_O_WORKDIR/logs/train_attention_fibrosis.out
#PBS -e $PBS_O_WORKDIR/logs/train_attention_fibrosis.err
: "${PROJECT_ROOT:?PROJECT_ROOT no está definida — exporta esta variable antes de lanzar el job}"
module load CUDA/12.1.1
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild
cd ${PROJECT_ROOT}/diffusion_model
echo "Training fibrosis (attention U-Net) | Start: $(date)"
python train.py --class_idx 3 \
    --epochs 1000 \
    --save_every 50 \
    --output_dir outputs_attention
echo "End: $(date)"
