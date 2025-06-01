#!/bin/bash
pyenv activate env

export TRANSFORMERS_CACHE="/scratch/qido00001/.cache/huggingface"
export HF_HOME="/scratch/qido00001/.cache/huggingface"

echo "Starting code generation with selected models..."
python generate_code.py --dataset both --num_samples 3 --model qwen,phi-4,llama

echo "All models completed!" 