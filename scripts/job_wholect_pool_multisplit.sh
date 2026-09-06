#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=08:00:00
#PBS -N wholect_pool_multisplit
#PBS -o $PBS_O_WORKDIR/logs/wholect_pool_multisplit.out
#PBS -e $PBS_O_WORKDIR/logs/wholect_pool_multisplit.err
: "${PROJECT_ROOT:?PROJECT_ROOT no está definida — exporta esta variable antes de lanzar el job}"
module load CUDA/12.1.1
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild
cd ${PROJECT_ROOT}/classifier_lib

# Generate the two missing splits (seed 0 already exists)
python ${PROJECT_ROOT}/wholect/scripts/build_wholect_fixed_split.py --seed 1
python ${PROJECT_ROOT}/wholect/scripts/build_wholect_fixed_split.py --seed 2

for split_seed in 0 1 2; do
  for cond in real real_aug real_synth; do
    for pool in mean max; do
      echo "=== split_seed=$split_seed condition=$cond pooling=$pool ==="
      python ${PROJECT_ROOT}/wholect/scripts/classification_wholect_pool.py \
          --condition $cond --pooling $pool --split-seed $split_seed --seed $split_seed
    done
  done
done
