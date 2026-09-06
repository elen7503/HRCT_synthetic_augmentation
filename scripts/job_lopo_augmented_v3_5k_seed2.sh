#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=12:00:00
#PBS -N lopo_augmented_v3_5k_seed2
#PBS -o $PBS_O_WORKDIR/logs/lopo_augmented_v3_5k_seed2.out
#PBS -e $PBS_O_WORKDIR/logs/lopo_augmented_v3_5k_seed2.err
: "${PROJECT_ROOT:?PROJECT_ROOT no está definida — exporta esta variable antes de lanzar el job}"

module load CUDA/12.1.1
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild
cd ${PROJECT_ROOT}/classifier_lib

echo "=========================================="
echo "Job: LOPO Augmented v3_5k (fixed synthetic dose, all real data kept, no class weighting)"
echo "Start: $(date)"
echo "=========================================="
python Supplementary_materials/classification_lopo_augmented_v3.py --seed 2
echo "=========================================="
echo "End: $(date)"
echo "=========================================="