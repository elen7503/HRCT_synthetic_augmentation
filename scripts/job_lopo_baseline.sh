#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=12:00:00
#PBS -N lopo_baseline
#PBS -o $PBS_O_WORKDIR/logs/lopo_baseline.out
#PBS -e $PBS_O_WORKDIR/logs/lopo_baseline.err
: "${PROJECT_ROOT:?PROJECT_ROOT no está definida — exporta esta variable antes de lanzar el job}"

module load CUDA/12.1.1
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild

cd ${PROJECT_ROOT}

echo "=========================================="
echo "Job: LOPO Baseline (real data only)"
echo "Start: $(date)"
echo "=========================================="

python ${PROJECT_ROOT}/patches/scripts/classification_lopo.py

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
