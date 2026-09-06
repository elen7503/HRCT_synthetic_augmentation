#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=04:00:00
#PBS -N ema_finetune_fibrosis
#PBS -o $PBS_O_WORKDIR/logs/ema_finetune_fibrosis.out
#PBS -e $PBS_O_WORKDIR/logs/ema_finetune_fibrosis.err
: "${PROJECT_ROOT:?PROJECT_ROOT no está definida — exporta esta variable antes de lanzar el job}"

module load CUDA/12.1.1
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild
cd ${PROJECT_ROOT}/diffusion_model

echo "=========================================="
echo "Job: EMA fine-tune, fibrosis"
echo "Start: $(date)"
echo "=========================================="
python train_ema_finetune.py --class_idx 3 \
    --weightfile outputs/fibrosis/best.pt \
    --output_dir outputs/fibrosis_ema \
    --extra_epochs 300
echo "=========================================="
echo "End: $(date)"
echo "=========================================="
