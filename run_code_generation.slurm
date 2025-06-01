#!/bin/bash
#SBATCH --job-name=code-generation
#SBATCH --output=logs/code_gen_%A_%a.log
#SBATCH --array=1-3%1  # Run one model at a time
#SBATCH --constraint=kine  # Use kine nodes which have better GPUs
#SBATCH --cpus-per-task=8  # More CPUs for data processing
#SBATCH --gres=gpu:1  # Request one GPU
#SBATCH --mem=160G  # Large memory for Llama-70B
#SBATCH --time=24:00:00  # 24 hours should be enough for all tasks

# Activate virtual environment if you have one
source /scratch/qido00001/smell/env/bin/Activate     

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE="/scratch/qido00001/.cache/huggingface"
export HF_HOME="/scratch/qido00001/.cache/huggingface"

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the script with different models based on array task ID
case $SLURM_ARRAY_TASK_ID in
    1)
        # Qwen model
        srun python generate_code.py --dataset both --num_samples 3 --model qwen
        ;;
    2)
        # Phi-4 model
        srun python generate_code.py --dataset both --num_samples 3 --model phi-4
        ;;
    3)
        # Llama-70B model
        srun python generate_code.py --dataset both --num_samples 3 --model llama
        ;;
esac 