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

# Source system profile to ensure the 'module' command is defined in non-interactive shell
if [ -f /etc/profile ]; then
    source /etc/profile
fi

# Try to source FSL directly from the project directory first
export FSLDIR=/sc-projects/sc-proj-cc15-sylt/fsl
if [ -d "$FSLDIR" ]; then
    export PATH=${FSLDIR}/bin:${PATH}
    if [ -f "${FSLDIR}/etc/fslconf/fsl.sh" ]; then
        source ${FSLDIR}/etc/fslconf/fsl.sh
    fi
else
    # Fallback to module load if the custom directory is not found
    module load fsl/6.0.7.18 || module load fsl || module load FSL || true
fi

# Ensure log directory exists
mkdir -p logs

# Set python path if needed
export PYTHONPATH=$(pwd)

# Pass all arguments directly to run_tumorsynth.py
python run_tumorsynth.py "$@"
