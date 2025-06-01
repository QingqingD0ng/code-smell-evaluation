#!/bin/bash
#SBATCH --job-name=code-generation
#SBATCH --output=/scratch/qido00001/smell/logs/code_gen_%j.log
#SBATCH --cpus-per-task=8  # More CPUs for data processing
#SBATCH --mem=400G  # Large memory for Llama-70B
#SBATCH --partition=anywhere

srun run.sh