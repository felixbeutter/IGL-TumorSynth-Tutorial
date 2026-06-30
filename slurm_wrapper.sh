#!/bin/bash
#SBATCH --job-name=tumorsynth
#SBATCH --partition=gpu
#SBATCH --nodes=1   
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

# Initialize conda environments
source ~/.bashrc
conda activate nnUNet_v1.7

# Source environment variables for nnUNet
source tools/tumorsynth/nnUNet_v1.7_path.sh

# Ensure log directory exists
mkdir -p logs

# Set python path if needed
export PYTHONPATH=$(pwd)

# Pass all arguments directly to run_tumorsynth.py
python run_tumorsynth.py "$@"
