#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=12:00:00
#PBS -N wholect_no_augment
#PBS -o $PBS_O_WORKDIR/logs/wholect_no_augment.out
#PBS -e $PBS_O_WORKDIR/logs/wholect_no_augment.err
: "${PROJECT_ROOT:?PROJECT_ROOT no está definida — exporta esta variable antes de lanzar el job}"
module load CUDA/12.1.1
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild
cd ${PROJECT_ROOT}/classifier_lib
python ${PROJECT_ROOT}/wholect/scripts/classification_wholect_lopo.py --seed 0 --no-augment