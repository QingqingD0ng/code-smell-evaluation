#!/bin/bash
#SBATCH --job-name=code-generation
#SBATCH --output=/scratch/qido00001/logs/log_%j.log
#SBATCH --cpus-per-task=2  
#SBATCH --mem=112G 
#SBATCH --partition=anywhere
#SBATCH --nodelist=aventinus
#SBATCH --time=1-00:00:00

# Default values
MODEL=${1:-"qwen"}
DATASET=${2:-"codereval"}
TECHNIQUE=${3:-"baseline"}
TASK_RANGE=${4:-"0-10"}
# Build the command with parameters
CMD="python generate_code.py --model $MODEL --dataset $DATASET --technique $TECHNIQUE --task_range $TASK_RANGE"

# Run the command
srun /scratch/qido00001/code-smell-evaluation/run.sh "$CMD"