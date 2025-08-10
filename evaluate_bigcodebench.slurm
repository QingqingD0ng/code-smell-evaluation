#!/bin/bash
#SBATCH --job-name=bigcodebench_eval
#SBATCH --output=/scratch/qido00001/logs/log_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=anywhere
#SBATCH --time=24:00:00
#SBATCH --array=0-14

# Build the command with parameters
CMD="python -m bigcodebench.evaluate --execution local --split complete --subset hard --samples ${bigcodebench_files[$SLURM_ARRAY_TASK_ID]} --no-gt --output-dir evaluation_results/${model_strategy}"

# Define the BigCodeBench sanitized files
declare -a bigcodebench_files=(
    "extracted_results_sanitized/bigcodebench/phi-3-bigcodebench-baseline-merged-sanitized.jsonl"
    "extracted_results_sanitized/bigcodebench/phi-3-bigcodebench-cot-merged-sanitized.jsonl"
    "extracted_results_sanitized/bigcodebench/phi-3-bigcodebench-persona-merged-sanitized.jsonl"
    "extracted_results_sanitized/bigcodebench/phi-3-bigcodebench-quality_focused-merged-sanitized.jsonl"
    "extracted_results_sanitized/bigcodebench/phi-3-bigcodebench-rci-merged-sanitized.jsonl"
    "extracted_results_sanitized/bigcodebench/phi-4-bigcodebench-baseline-merged-sanitized.jsonl"
    "extracted_results_sanitized/bigcodebench/phi-4-bigcodebench-cot-merged-sanitized.jsonl"
    "extracted_results_sanitized/bigcodebench/phi-4-bigcodebench-persona-merged-sanitized.jsonl"
    "extracted_results_sanitized/bigcodebench/phi-4-bigcodebench-quality_focused-merged-sanitized.jsonl"
    "extracted_results_sanitized/bigcodebench/phi-4-bigcodebench-rci-merged-sanitized.jsonl"
    "extracted_results_sanitized/bigcodebench/qwen-bigcodebench-baseline-merged-sanitized.jsonl"
    "extracted_results_sanitized/bigcodebench/qwen-bigcodebench-cot-merged-sanitized.jsonl"
    "extracted_results_sanitized/bigcodebench/qwen-bigcodebench-persona-merged-sanitized.jsonl"
    "extracted_results_sanitized/bigcodebench/qwen-bigcodebench-quality_focused-merged-sanitized.jsonl"
    "extracted_results_sanitized/bigcodebench/qwen-bigcodebench-rci-merged-sanitized.jsonl"
)

# Get the current file based on array index
current_file="${bigcodebench_files[$SLURM_ARRAY_TASK_ID]}"

# Extract model and strategy from filename for output naming
filename=$(basename "$current_file" .jsonl)
model_strategy=$(echo "$filename" | sed 's/-merged-sanitized//')

# Run the command
srun /scratch/qido00001/code-smell-evaluation/evaluate_bigcodebench.sh "$CMD"
