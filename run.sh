#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Activate the Python environment
pyenv activate env

# Set cache directories
export TRANSFORMERS_CACHE="/scratch/qido00001/.cache/huggingface"
export HF_HOME="/scratch/qido00001/.cache/huggingface"

# Create cache directories if they do not exist
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$HF_HOME"

echo "Starting code generation with selected models..."
python generate_code.py --dataset both --num_samples 3 --model qwen,phi-4,llama

echo "All models completed!"
