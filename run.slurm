#!/bin/bash
#SBATCH --job-name=code-generation
#SBATCH --output=/scratch/qido00001/smell/logs/code_gen_%j.log
#SBATCH --cpus-per-task=8  
#SBATCH --mem=400G  
#SBATCH --partition=anywhere
#SBATCH --nodelist=maxl19
#SBATCH --time=1-00:00:00
srun ./run.sh