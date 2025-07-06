#!/bin/bash
#SBATCH --job-name=code-generation
#SBATCH --output=/scratch/qido00001/logs/log_%j.log
#SBATCH --cpus-per-task=2  
#SBATCH --mem=112G 
#SBATCH --partition=anywhere
#SBATCH --nodelist=aventinus
#SBATCH --time=1-00:00:00

# Default values
DATASET=${1:-"codereval"}
TASK_RANGE=${2:-"0-10"}
# Build the command with parameters
CMD="python generate_code.py --model qwen --dataset $DATASET --technique rci --task_range $TASK_RANGE"

# Run the command
srun /scratch/qido00001/code-smell-evaluation/run.sh "$CMD"