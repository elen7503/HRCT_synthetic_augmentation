#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -N sample_lungddpm_test
#PBS -o $PBS_O_WORKDIR/logs/sample_lungddpm_test.out
#PBS -e $PBS_O_WORKDIR/logs/sample_lungddpm_test.err
: "${PROJECT_ROOT:?PROJECT_ROOT no está definida — exporta esta variable antes de lanzar el job}"
module load CUDA/12.1.1
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate nnUnet_ILD
cd ${PROJECT_ROOT}/wholect/lung_ddpm_model/Lung-DDPM
CUDA_VISIBLE_DEVICES=0 python -u sample_ild_2d_balanced.py \
  --data-root ../ILD_dataset \
  --split-report ../nnUNet_raw/Dataset502_ILD/conversion_report.json \
  --checkpoint checkpoints/ILD-DDPM-2D-V3/ild_ddpm_2d_step_0014000.pth \
  --output-dir generated/ILD-DDPM-2D-V3-balanced-test \
  --target-per-class 250 \
  --guidance-scale 1.5 \
  --blend-mode lung \
  --inpainting-strength 0.35
