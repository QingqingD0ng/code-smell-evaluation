#!/bin/bash
#SBATCH --job-name=code-generation
#SBATCH --output=/scratch/qido00001/smell/logs/code_gen_%j.log
#SBATCH --cpus-per-task=8  # More CPUs for data processing
#SBATCH --mem=400G  # Large memory for Llama-70B
#SBATCH --partition=anywhere

# Activate virtual environment if you have one
pyenv activate env    

# Set environment variables
export TRANSFORMERS_CACHE="/scratch/qido00001/.cache/huggingface"
export HF_HOME="/scratch/qido00001/.cache/huggingface"

# Run all models sequentially
echo "Starting code generation with selected models..."
srun python generate_code.py --dataset both --num_samples 3 --model qwen,phi-4,llama

echo "All models completed!" 