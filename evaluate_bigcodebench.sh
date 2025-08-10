#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Activate the Python environment
source /scratch/qido00001/.bashrc
pyenv global 3.10.12

echo "Starting BigCodeBench evaluation..."
# Execute the command passed from SLURM script
eval "$1"

echo "BigCodeBench evaluation completed!"
