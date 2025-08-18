#!/bin/bash
#SBATCH --job-name=bigcodebench_eval
#SBATCH --output=/scratch/qido00001/logs/log_%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=anywhere
#SBATCH --time=24:00:00
#SBATCH --array=0-14

# The command to run. Pass the array id to the shell script
srun /scratch/qido00001/code-smell-evaluation/evaluate_bigcodebench.sh $SLURM_ARRAY_TASK_ID
