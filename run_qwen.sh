#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Activate the Python environment
source /scratch/qido00001/.bashrc
pyenv activate env

# Clear cache
rm -rf ~/.cache/*

# Set cache directories
export TRANSFORMERS_CACHE="/scratch/qido00001/.cache/huggingface"
export HF_HOME="/scratch/qido00001/.cache/huggingface"

# Create cache directories if they do not exist
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$HF_HOME"

echo "Starting code generation with selected models..."
python generate_code.py --dataset both --model qwen

echo "Qwen model completed!"
