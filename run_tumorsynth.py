import argparse, shutil, ants
import numpy as np
from pathlib import Path

from registration import register_to_atlas, transform_to_native, transform_to_atlas
from segmentation import execute_tumorsynth, extract_whole_tumor_mask, extract_sub_masks
from utils import log


INNER_TUMOR_LABELS = {
    'necrosis': 1,
    'enhancing_tumor': 4,
    'edema': 2,
}


def process_scan(input_paths: list[Path], reg_input_path: Path, scan_name: str, output_dir: Path, atlas_dir: Path, use_gpu: bool = False, cleanup: bool = False):
    """
    Pipeline: Processes multi-modal MRI scans through the complete TumorSynth pipeline.

    Args:
        input_paths: List of paths to the input native NIfTI scans (e.g., T1c, T2, FLAIR).
                     Must be co-registered to each other.
        reg_input_path: The specific input file to use for atlas registration.
        scan_name: Base name for output files and directory.
        output_dir: Path to the base directory where results will be stored.
        atlas_dir: Path to the SRI24 atlas templates directory.
        use_gpu: If True, uses GPU for segmentation.
        cleanup: If True, deletes intermediate atlas-space files and transforms.

    Returns:
        None
    """
    # Prepare the dedicated output directory.
    scan_out_dir = output_dir / scan_name
    scan_out_dir.mkdir(parents=True, exist_ok=True)

    # Define paths to the SRI24 atlas reference templates.
    sri24_t1 = atlas_dir / 'SRI24_T1.nii.gz'
    sri24_mask = atlas_dir / 'SRI24_brain_mask.nii.gz'
    
    log(f"[{scan_name}] 1. Registering to SRI24 Atlas using {reg_input_path.name}... ", newline=False)
    # Load the moving image (registration modality) and fixed images (atlas T1 and brain mask).
    reg_native_img = ants.image_read(str(reg_input_path))
    atlas_img = ants.image_read(str(sri24_t1))
    atlas_mask = ants.image_read(str(sri24_mask))
    
    # Perform SyN registration on the primary modality
    reg_in_atlas_img, registration = register_to_atlas(reg_native_img, atlas_img, atlas_mask)
    log("done.", timestamp=False)

    log(f"[{scan_name}] 2. Transforming all modalities to Atlas space... ")
    atlas_paths = []
    roi_paths = []
    
    for ipath in input_paths:
        modality_name = ipath.name.split('.nii')[0]
        # If it's the registration modality, we already have it transformed and skull-stripped
        if ipath == reg_input_path:
            mod_in_atlas_img = reg_in_atlas_img
        else:
            mod_native_img = ants.image_read(str(ipath))
            # Transform to atlas space using forward transforms
            mod_in_atlas_raw = transform_to_atlas(atlas_img, mod_native_img, registration['fwdtransforms'])
            # Skull strip using atlas mask
            mod_in_atlas_img = mod_in_atlas_raw * atlas_mask
            
        # Save skull-stripped atlas-space image
        mod_atlas_path = scan_out_dir / f"{modality_name}_SRI24.nii.gz"
        ants.image_write(mod_in_atlas_img, str(mod_atlas_path))
        atlas_paths.append(mod_atlas_path)
        log(f"  - Saved {mod_atlas_path.name}")
    
    log(f"[{scan_name}] 3. Running Whole Tumor Segmentation... ", newline=False)
    # Run TumorSynth in 'wholetumor' mode on all modalities
    wt_raw_atlas_path = scan_out_dir / "tumorsynth_wt_raw_SRI24.nii.gz"
    execute_tumorsynth(atlas_paths, wt_raw_atlas_path, mask_type='wholetumor', use_gpu=use_gpu)
    log("done.", timestamp=False)
    
    # Extract the binary whole tumor mask
    wt_atlas_img = ants.image_read(str(wt_raw_atlas_path))
    wt_mask_atlas_img = extract_whole_tumor_mask(wt_atlas_img)
    ants.image_write(wt_mask_atlas_img, str(scan_out_dir / f"{scan_name}_whole_tumor_mask_SRI24.nii.gz"))
    
    log(f"[{scan_name}] 4. Preparing ROIs and Running Inner Tumor Segmentation... ", newline=False)
    # Mask all atlas-aligned images with the binary whole tumor mask to isolate the tumor ROI.
    for ipath, mod_atlas_path in zip(input_paths, atlas_paths):
        modality_name = ipath.name.split('.nii')[0]
        mod_atlas_img = ants.image_read(str(mod_atlas_path))
        roi_img = mod_atlas_img * wt_mask_atlas_img
        roi_path = scan_out_dir / f"{modality_name}_roi_SRI24.nii.gz"
        ants.image_write(roi_img, str(roi_path))
        roi_paths.append(roi_path)
    
    # Run TumorSynth in 'innertumor' mode
    it_raw_atlas_path = scan_out_dir / "tumorsynth_it_raw_SRI24.nii.gz"
    execute_tumorsynth(roi_paths, it_raw_atlas_path, mask_type='innertumor', use_gpu=use_gpu)
    it_atlas_img = ants.image_read(str(it_raw_atlas_path))
    log("done.", timestamp=False)
    
    log(f"[{scan_name}] 5. Transforming results back to Native Space... ", newline=False)
    inv_transforms = registration['invtransforms']
    
    # Transform masks back to native space (using reg_native_img as reference geometry)
    wt_mask_native_img = transform_to_native(reg_native_img, wt_mask_atlas_img, inv_transforms)
    ants.image_write(wt_mask_native_img, str(scan_out_dir / f"{scan_name}_whole_tumor_mask.nii.gz"))
    
    it_native_img = transform_to_native(reg_native_img, it_atlas_img, inv_transforms)
    
    # Remap labels
    it_arr = it_native_img.numpy()
    remapped_arr = np.zeros_like(it_arr)
    raw_to_class = {1: 'edema', 2: 'enhancing_tumor', 3: 'necrosis'}
    
    for raw_val, class_name in raw_to_class.items():
        target_val = INNER_TUMOR_LABELS.get(class_name)
        if target_val is not None:
            remapped_arr[it_arr == raw_val] = target_val
            
    it_native_img_remapped = ants.from_numpy(
        remapped_arr.astype(np.float32),
        origin=it_native_img.origin,
        spacing=it_native_img.spacing,
        direction=it_native_img.direction
    )
    
    ants.image_write(it_native_img_remapped, str(scan_out_dir / f"{scan_name}_inner_tumor_seg.nii.gz"))
    extract_sub_masks(it_native_img_remapped, scan_out_dir, scan_name, INNER_TUMOR_LABELS)
    log("done.", timestamp=False)
        
    if cleanup:
        log(f"[{scan_name}] 6. Cleaning up intermediate files... ", newline=False)
        paths_to_remove = atlas_paths + roi_paths + [
            wt_raw_atlas_path,
            scan_out_dir / f"{scan_name}_whole_tumor_mask_SRI24.nii.gz",
            it_raw_atlas_path
        ]
        for p in paths_to_remove:
            if p.exists(): p.unlink()
            
        for t_list in [registration.get('fwdtransforms', []), registration.get('invtransforms', [])]:
            for t_path in t_list:
                p = Path(t_path)
                if p.exists(): p.unlink()
        log("done.", timestamp=False)

    log(f"[{scan_name}] Processing complete! Outputs saved in {scan_out_dir}")

def main():
    """
    Parses command line arguments and runs process_scan.
    """
    parser = argparse.ArgumentParser(description="Run TumorSynth pipeline on multi-modal native NIfTI scans.")
    parser.add_argument("-i", "--inputs", type=Path, nargs='+', required=True, 
                        help="Path to the input native NIfTI scans (e.g., T1c, T2, FLAIR). Must be co-registered.")
    parser.add_argument("--reg-input", type=Path, default=None,
                        help="The input file to use for atlas registration. Defaults to the first input.")
    parser.add_argument("--scan-name", type=str, default=None,
                        help="Base name for the outputs. Defaults to the name of the reg-input without extension.")
    parser.add_argument("-o", "--output-dir", type=Path, default=None, 
                        help="Directory to save the outputs. Defaults to the same directory as the reg-input scan.")
    parser.add_argument("-a", "--atlas-dir", type=Path, default=None,
                        help="Path to the SRI24 atlas directory. Defaults to assets/SRI24_atlas relative to this script.")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run TumorSynth using GPU")
    parser.add_argument("-c", "--cleanup", action="store_true", help="Delete intermediate files, keeping only final native-space segmentations")
    args = parser.parse_args()

    # Check for FSL dependencies upfront to avoid failing after the long registration step.
    missing_deps = []
    for dep in ('fslmaths', 'fslmerge'):
        if not shutil.which(dep):
            missing_deps.append(dep)
    if missing_deps:
        log("Error: Missing required FSL (FMRIB Software Library) dependencies: " + ", ".join(missing_deps))
        log("Please ensure FSL is installed and available in your PATH.")
        log("On a cluster, you may need to run 'module load fsl' or source your FSL configuration.")
        return

    # Resolve input paths
    input_paths = [p.resolve() for p in args.inputs]
    for p in input_paths:
        if not p.exists():
            log(f"Error: Input file {p} does not exist.")
            return
            
    # Resolve reg_input
    if args.reg_input:
        reg_input_path = args.reg_input.resolve()
        if reg_input_path not in input_paths:
            log(f"Error: reg-input {reg_input_path} must be one of the files provided in --inputs.")
            return
    else:
        reg_input_path = input_paths[0]

    # Determine scan name and output directory
    scan_name = args.scan_name if args.scan_name else reg_input_path.name.split('.nii')[0]
    out_dir = args.output_dir.resolve() if args.output_dir else reg_input_path.parent
    
    # Locate the atlas templates folder, defaulting to the assets/SRI24_atlas directory in this repo.
    base_dir = Path(__file__).parent.absolute()
    atlas_dir = args.atlas_dir.resolve() if args.atlas_dir else base_dir / 'assets' / 'SRI24_atlas'

    if not atlas_dir.exists():
        log(f"Error: Atlas directory {atlas_dir} does not exist.")
        return

    # Run the pipeline on the specified scans.
    process_scan(input_paths, reg_input_path, scan_name, out_dir, atlas_dir, use_gpu=args.gpu, cleanup=args.cleanup)

if __name__ == "__main__":
    main()
