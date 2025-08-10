#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Get the array ID from the first argument
ARRAY_ID=$1

# Activate the Python environment
source /scratch/qido00001/.bashrc
pyenv global 3.10.12

echo "Starting BigCodeBench evaluation for array ID: $ARRAY_ID"

# Define the BigCodeBench sanitized files
declare -a bigcodebench_files=(
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
current_file="${bigcodebench_files[$ARRAY_ID]}"

# Extract model and strategy from filename for output naming
filename=$(basename "$current_file" .jsonl)
model_strategy=$(echo "$filename" | sed 's/-merged-sanitized//')

echo "Processing file: $current_file"
echo "Model/Strategy: $model_strategy"

# Run the BigCodeBench evaluation
bigcodebench.evaluate \
    --execution local \
    --split complete \
    --subset hard \
    --samples "$current_file" \
    --no-gt \
    --output-dir "evaluation_results/$model_strategy"

echo "BigCodeBench evaluation completed for: $model_strategy"
echo "Results saved to: evaluation_results/$model_strategy"
