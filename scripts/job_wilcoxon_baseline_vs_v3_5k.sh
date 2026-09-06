#!/bin/bash
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=00:10:00
#PBS -N wilcoxon_baseline_vs_v3_5k
#PBS -o $PBS_O_WORKDIR/patches/logs/wilcoxon_baseline_vs_v3_5k.out
#PBS -e $PBS_O_WORKDIR/patches/logs/wilcoxon_baseline_vs_v3_5k.err
: "${PROJECT_ROOT:?PROJECT_ROOT no está definida — exporta esta variable antes de lanzar el job}"
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild
cd ${PROJECT_ROOT}
echo "Job: Wilcoxon signed-rank, baseline vs v3_5k (fixed-dose 5000/class), patch-level LOPO"
echo "Start: $(date)"
python ${PROJECT_ROOT}/patches/scripts/wilcoxon_baseline_vs_v3_5k.py
echo "End: $(date)"
