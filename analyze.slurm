#!/bin/bash
#SBATCH --job-name=code-analysis
#SBATCH --output=/scratch/qido00001/logs/log_%j.log
#SBATCH --cpus-per-task=2  
#SBATCH --mem=20G 
#SBATCH --partition=anywhere
#SBATCH --time=1-00:00:00


# Build the command with parameters
CMD="python analyze_code_smells.py"

# Run the command
srun /scratch/qido00001/code-smell-evaluation/analyze.sh "$CMD"