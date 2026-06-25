import subprocess
import ants
from pathlib import Path
from config import TUMORSYNTH_DIR, USE_GPU, INNER_TUMOR_LABELS

def execute_tumorsynth(input_img: Path, out_img: Path, mask_type: str = 'wholetumor'):
    """
    Runs the mri_TumorSynth command line tool.
    
    Args:
        input_img: Path to the input NIfTI image.
        out_img: Path where the output mask will be saved.
        mask_type: Either 'wholetumor' or 'innertumor'.
    """
    if mask_type not in ('wholetumor', 'innertumor'):
        raise ValueError("mask_type must be 'wholetumor' or 'innertumor'")

    script = TUMORSYNTH_DIR / 'mri_TumorSynth'
    if not script.exists():
        raise FileNotFoundError(
            f"Could not find 'mri_TumorSynth' script at {script}. "
            "Please check the TUMORSYNTH_DIR path in config.py"
        )

    cmd = [
        str(script),
        '--i', str(input_img),
        '--o', str(out_img),
        '--threads', '-1',
        '--nnUnet', str(TUMORSYNTH_DIR)
    ]
    
    if not USE_GPU:
        cmd.append('--cpu')
    
    if mask_type == 'wholetumor':
        cmd.append('--wholetumor')
    elif mask_type == 'innertumor':
        cmd.append('--innertumor')

    print(f"Running TumorSynth ({mask_type}) on {input_img.name}...")
    subprocess.run(cmd, check=True)

def extract_whole_tumor_mask(wt_atlas_img):
    """
    Extracts the binary whole tumor mask from the 18-label parcellation.
    Label 18 corresponds to the whole tumor.
    """
    return ants.threshold_image(wt_atlas_img, low_thresh=17.5, high_thresh=18.5)

def extract_sub_masks(it_native_img, scan_out_dir: Path, scan_name: str):
    """
    Extracts individual BraTS-compliant sub-masks from the multi-label inner tumor segmentation
    and saves them to the disk.
    """
    for name, label in INNER_TUMOR_LABELS.items():
        mask_native_img = ants.threshold_image(
            it_native_img, low_thresh=label - 0.5, high_thresh=label + 0.5
        )
        out_path = scan_out_dir / f"{scan_name}_{name}_mask.nii.gz"
        ants.image_write(mask_native_img, str(out_path))
