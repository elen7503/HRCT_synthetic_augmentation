#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -N ddpm_train
#PBS -o /rds/general/user/eh1121/home/Final_Project/diffusion_model/logs/job_${CLASS_IDX}.out
#PBS -e /rds/general/user/eh1121/home/Final_Project/diffusion_model/logs/job_${CLASS_IDX}.err

# ── Load modules ──────────────────────────────────────────────────────────────
module load CUDA/12.1.1

# ── Activate conda env ────────────────────────────────────────────────────────
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild

# ── Move to working directory ─────────────────────────────────────────────────
cd /rds/general/user/eh1121/home/Final_Project/diffusion_model

# ── Class index passed via qsub -v CLASS_IDX=N ───────────────────────────────
CLASS_IDX=${CLASS_IDX:-3}

echo "=========================================="
echo "Job ID:     $PBS_JOBID"
echo "Node:       $HOSTNAME"
echo "Class idx:  $CLASS_IDX"
echo "Start time: $(date)"
echo "=========================================="

nvidia-smi

python train.py \
    --class_idx  $CLASS_IDX \
    --epochs     3000 \
    --batch_size 64 \
    --lr         2e-4 \
    --timesteps  1000 \
    --save_every 500

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
