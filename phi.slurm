#!/bin/bash
#SBATCH --job-name=code-generation
#SBATCH --output=/scratch/qido00001/logs/log_%j.log
#SBATCH --cpus-per-task=2  
#SBATCH --mem=32G  
#SBATCH --partition=anywhere
#SBATCH --nodelist=aventinus
#SBATCH --time=1-00:00:00
srun ./scratch/qido00001/code-smell-evaluation/run_qwen.sh