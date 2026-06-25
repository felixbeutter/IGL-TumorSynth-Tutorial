#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export nnUNet_raw_data_base="$SCRIPT_DIR/nnUNet_v1.7/nnUNet_raw_data_base"
export nnUNet_preprocessed="$SCRIPT_DIR/nnUNet_v1.7/nnUNet_preprocessed"
export RESULTS_FOLDER="$SCRIPT_DIR/nnUNet_v1.7/nnUNet_trained_models"

