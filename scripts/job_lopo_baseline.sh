#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=12:00:00
#PBS -N lopo_baseline
#PBS -o /rds/general/user/eh1121/home/Final_Project/logs/lopo_baseline.out
#PBS -e /rds/general/user/eh1121/home/Final_Project/logs/lopo_baseline.err

module load CUDA/12.1.1
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild

cd /rds/general/user/eh1121/home/Final_Project

echo "=========================================="
echo "Job: LOPO Baseline (real data only)"
echo "Start: $(date)"
echo "=========================================="

python ILD-Segmentation-And-Classification-DL/Supplementary_materials/classification_lopo.py

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
