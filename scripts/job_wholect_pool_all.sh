#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=03:00:00
#PBS -N wholect_pool_all
#PBS -o $PBS_O_WORKDIR/logs/wholect_pool_all.out
#PBS -e $PBS_O_WORKDIR/logs/wholect_pool_all.err
: "${PROJECT_ROOT:?PROJECT_ROOT no está definida — exporta esta variable antes de lanzar el job}"
module load CUDA/12.1.1
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild
cd ${PROJECT_ROOT}/classifier_lib

for cond in real real_aug real_synth; do
  for pool in mean max; do
    echo "=== condition=$cond pooling=$pool ==="
    python ${PROJECT_ROOT}/wholect/scripts/classification_wholect_pool.py \
        --condition $cond --pooling $pool --split-seed 0 --seed 0
  done
done
