import subprocess
import ants
from pathlib import Path
from config import TUMORSYNTH_DIR, INNER_TUMOR_LABELS

import os
from utils import run_cmd, log

def execute_tumorsynth(input_img: Path, out_img: Path, mask_type: str = 'wholetumor', use_gpu: bool = False):
    """
    Step 2/4: Runs the mri_TumorSynth command line tool for segmentation.

    Args:
        input_img: Path to the input NIfTI image (T1c in atlas or ROI in atlas).
        out_img: Path where the raw segmentation/parcellation mask will be saved.
        mask_type: Either 'wholetumor' or 'innertumor' (default: 'wholetumor').
        use_gpu: If True, uses GPU acceleration. If False, runs on CPU.

    Returns:
        None
    """
    # Verify the requested task mode is supported.
    if mask_type not in ('wholetumor', 'innertumor'):
        raise ValueError("mask_type must be 'wholetumor' or 'innertumor'")

    # Locate the mri_TumorSynth binary/shell wrapper in the local tools/ directory.
    script = TUMORSYNTH_DIR / 'mri_TumorSynth'
    if not script.exists():
        raise FileNotFoundError(
            f"Could not find 'mri_TumorSynth' script at {script}. "
            "Please check the TUMORSYNTH_DIR path in config.py"
        )

    # Query the physical/logical CPU count to limit thread allocation.
    # Passing -1 or too many threads can cause PyTorch/OMP runtime errors.
    num_threads = str(os.cpu_count() or 4)

    # Build the list of arguments to pass to the TumorSynth command.
    cmd = [
        str(script),
        '--i', str(input_img),
        '--o', str(out_img),
        '--threads', num_threads,
        '--nnUnet', str(TUMORSYNTH_DIR)
    ]
    
    # Configure processing unit (GPU vs CPU).
    if not use_gpu:
        cmd.append('--cpu')
    
    # Select the model checkpoint/task to run.
    if mask_type == 'wholetumor':
        cmd.append('--wholetumor')
    elif mask_type == 'innertumor':
        cmd.append('--innertumor')

    log(f"Running TumorSynth ({mask_type}) on {input_img.name}... ", newline=False)
    # Execute the subprocess command using our unified run_cmd utility.
    run_cmd(cmd)
    log("done.", timestamp=False)

def extract_whole_tumor_mask(wt_atlas_img):
    """
    Step 3: Extract binary whole tumor mask from 18-label parcellation.

    Args:
        wt_atlas_img: ANTsImage of the 18-label whole tumor parcellation in atlas space.

    Returns:
        ants.ANTsImage: Binary whole tumor mask image in atlas space.
    """
    return ants.threshold_image(wt_atlas_img, low_thresh=17.5, high_thresh=18.5)

def extract_sub_masks(it_native_img, scan_out_dir: Path, scan_name: str):
    """
    Step 6: Extract individual BraTS-compliant sub-masks from inner tumor segmentation.

    Args:
        it_native_img: ANTsImage of the inner tumor segmentation in native space.
        scan_out_dir: Path to the output directory where sub-masks will be saved.
        scan_name: Base name of the scan for output file naming.

    Returns:
        None
    """
    # Iterate over each predefined inner tumor sub-structure (necrosis, enhancing, etc.)
    for name, label in INNER_TUMOR_LABELS.items():
        # Apply a narrow threshold around the integer label to extract the binary region.
        mask_native_img = ants.threshold_image(
            it_native_img, low_thresh=label - 0.5, high_thresh=label + 0.5
        )
        # Write the resulting binary mask to the patient's output directory.
        out_path = scan_out_dir / f"{scan_name}_{name}_mask.nii.gz"
        ants.image_write(mask_native_img, str(out_path))
