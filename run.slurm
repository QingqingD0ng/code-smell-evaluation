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
TECHNIQUE=${3:-"baseline"}

# Build the command with parameters
CMD="python generate_code.py --model $MODEL --dataset $DATASET --technique $TECHNIQUE"

# Add baseline file if technique is rci
if [ "$TECHNIQUE" = "rci" ]; then
    if [ -z "$BASELINE_FILE" ]; then
        echo "Error: Baseline file is required for RCI technique"
        exit 1
    fi
    CMD="$CMD --baseline_file $BASELINE_FILE"
fi

# Run the command
srun /scratch/qido00001/code-smell-evaluation/run.sh "$CMD"