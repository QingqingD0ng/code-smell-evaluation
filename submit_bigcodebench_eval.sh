#!/bin/bash

# Script to submit individual SLURM jobs for BigCodeBench evaluation
# This approach gives more control and easier debugging

# Define the BigCodeBench sanitized files
bigcodebench_files=(
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

echo "Submitting BigCodeBench evaluation jobs for ${#bigcodebench_files[@]} files..."

# Submit a job for each file
for file in "${bigcodebench_files[@]}"; do
    # Extract model and strategy from filename
    filename=$(basename "$file" .jsonl)
    model_strategy=$(echo "$filename" | sed 's/-merged-sanitized//')
    
    echo "Submitting job for: $model_strategy"
    
    # Submit the job using sbatch
    job_id=$(sbatch \
        --job-name="eval_${model_strategy}" \
        --output="/scratch/qido00001/logs/log_%j.log" \
        --time=24:00:00 \
        --mem=32G \
        --cpus-per-task=4 \
        --partition=anywhere \
        --wrap="
            source /scratch/qido00001/.bashrc;
            pyenv global 3.10.12;
            echo 'Starting evaluation for: $file';
            echo 'Model/Strategy: $model_strategy';
            python -m bigcodebench.evaluate \
                --execution local \
                --split complete \
                --subset hard \
                --samples '$file' \
                --no-gt \
                --output-dir 'evaluation_results/$model_strategy';
            echo 'Evaluation completed for: $file';
        ")
    
    # Extract the job ID from sbatch output
    job_id=$(echo "$job_id" | grep -o '[0-9]\+')
    echo "  Submitted job ID: $job_id for $model_strategy"
done

echo "All BigCodeBench evaluation jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "Monitor logs in /scratch/qido00001/logs/"
