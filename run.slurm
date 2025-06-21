#!/bin/bash
#SBATCH --job-name=code-generation
#SBATCH --output=/scratch/qido00001/logs/log_%j.log
#SBATCH --cpus-per-task=2  
#SBATCH --mem=112G 
#SBATCH --partition=anywhere
#SBATCH --nodelist=aventinus
#SBATCH --time=1-00:00:00

# Default values
MODEL=${1:-"phi-3"}
DATASET=${2:-"codereval"}
BASELINE_FILE=${3:-"baseline"}

# Build the command with parameters
CMD="python analyze_code_smells.py --model $MODEL --dataset $DATASET --baseline_file $BASELINE_FILE --technique rci"

# Run the command
srun /scratch/qido00001/code-smell-evaluation/run.sh "$CMD"