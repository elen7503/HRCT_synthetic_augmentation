#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -N sweep_sample_attention_healthy
#PBS -o $PBS_O_WORKDIR/logs/sweep_sample_attention_healthy.out
#PBS -e $PBS_O_WORKDIR/logs/sweep_sample_attention_healthy.err
: "${PROJECT_ROOT:?PROJECT_ROOT no está definida — exporta esta variable antes de lanzar el job}"
module load CUDA/12.1.1
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild
cd ${PROJECT_ROOT}/diffusion_model
python sweep_sample.py --class_idx 0 --checkpoint_dir outputs_attention/healthy \
    --output_dir outputs_attention/checkpoint_sweep/healthy --n_samples 100
