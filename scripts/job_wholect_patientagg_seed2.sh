#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=02:30:00
#PBS -N wholect_patientagg_s2
#PBS -o $PBS_O_WORKDIR/logs/wholect_patientagg_seed2.out
#PBS -e $PBS_O_WORKDIR/logs/wholect_patientagg_seed2.err
: "${PROJECT_ROOT:?PROJECT_ROOT no está definida — exporta esta variable antes de lanzar el job}"
module load CUDA/12.1.1
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild
cd ${PROJECT_ROOT}/classifier_lib

echo "=== BASELINE seed2 ==="
python ${PROJECT_ROOT}/wholect/scripts/classification_wholect_patientagg_lopo.py \
    --data-dir ${PROJECT_ROOT}/ILD_DB_wholect_uint8 --seed 2

echo "=== AUGMENTED seed2 ==="
python ${PROJECT_ROOT}/wholect/scripts/classification_wholect_patientagg_lopo.py \
    --data-dir ${PROJECT_ROOT}/ILD_DB_wholect_augmented --seed 2
