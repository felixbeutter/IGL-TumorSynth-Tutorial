import argparse
from pathlib import Path
import ants

from registration import register_to_atlas, transform_to_native
from segmentation import execute_tumorsynth, extract_whole_tumor_mask, extract_sub_masks
from utils import log

def process_scan(native_t1c_path: Path, output_dir: Path, atlas_dir: Path, use_gpu: bool = False, cleanup: bool = False):
    """
    Pipeline: Processes a single T1c scan through the complete TumorSynth pipeline.

    Args:
        native_t1c_path: Path to the input native T1c NIfTI scan.
        output_dir: Path to the base directory where results will be stored.
        atlas_dir: Path to the SRI24 atlas templates directory.
        use_gpu: If True, uses GPU for segmentation.
        cleanup: If True, deletes intermediate atlas-space files and transforms.

    Returns:
        None
    """
    # Parse scan base name safely (e.g. 'UCSF-PDGM-0004_T1c' from 'UCSF-PDGM-0004_T1c.nii.gz')
    # and prepare the dedicated output directory.
    scan_name = native_t1c_path.name.split('.nii')[0]
    scan_out_dir = output_dir / scan_name
    scan_out_dir.mkdir(parents=True, exist_ok=True)

    # Define paths to the SRI24 atlas reference templates.
    # TumorSynth is trained on SRI24-aligned brain scans, so registration is mandatory.
    sri24_t1 = atlas_dir / 'SRI24_T1.nii.gz'
    sri24_mask = atlas_dir / 'SRI24_brain_mask.nii.gz'
    
    log(f"[{scan_name}] 1. Registering to SRI24 Atlas... ", newline=False)
    # Load the moving (native T1c) and fixed (atlas T1 and brain mask) images into ANTs.
    native_img = ants.image_read(str(native_t1c_path))
    atlas_img = ants.image_read(str(sri24_t1))
    atlas_mask = ants.image_read(str(sri24_mask))
    
    # Perform SyN registration, skull-strip the resulting image, and write to disk.
    # We save to disk because the TumorSynth CLI expects file paths as inputs.
    t1c_in_atlas_img, registration = register_to_atlas(native_img, atlas_img, atlas_mask)
    t1c_in_atlas_path = scan_out_dir / f"{scan_name}_t1c_SRI24.nii.gz"
    ants.image_write(t1c_in_atlas_img, str(t1c_in_atlas_path))
    
    log("done.", timestamp=False)
    log(f"[{scan_name}] 2. Running Whole Tumor Segmentation... ", newline=False)
    # Run TumorSynth in 'wholetumor' mode. This generates a raw multi-label parcellation.
    wt_raw_atlas_path = scan_out_dir / "tumorsynth_wt_raw_SRI24.nii.gz"
    execute_tumorsynth(t1c_in_atlas_path, wt_raw_atlas_path, mask_type='wholetumor', use_gpu=use_gpu)
    log("done.", timestamp=False)
    
    # Extract the binary whole tumor mask from the parcellation by thresholding label 18.
    # Save the mask in atlas space for downstream masking and potential verification.
    wt_atlas_img = ants.image_read(str(wt_raw_atlas_path))
    wt_mask_atlas_img = extract_whole_tumor_mask(wt_atlas_img)
    ants.image_write(wt_mask_atlas_img, str(scan_out_dir / f"{scan_name}_whole_tumor_mask_SRI24.nii.gz"))
    
    log(f"[{scan_name}] 3. Running Inner Tumor Segmentation... ", newline=False)
    # Mask the atlas-aligned T1c image with the binary whole tumor mask to isolate the tumor ROI.
    # The inner tumor segmentation model requires this masked ROI as input to focus on internal sub-structures.
    t1c_roi_img = t1c_in_atlas_img * wt_mask_atlas_img
    t1c_roi_path = scan_out_dir / f"{scan_name}_t1c_roi_SRI24.nii.gz"
    ants.image_write(t1c_roi_img, str(t1c_roi_path))
    
    # Run TumorSynth in 'innertumor' mode on the masked ROI to segment inner structures.
    it_raw_atlas_path = scan_out_dir / "tumorsynth_it_raw_SRI24.nii.gz"
    execute_tumorsynth(t1c_roi_path, it_raw_atlas_path, mask_type='innertumor', use_gpu=use_gpu)
    it_atlas_img = ants.image_read(str(it_raw_atlas_path))
    log("done.", timestamp=False)
    
    log(f"[{scan_name}] 4. Transforming results back to Native Space... ", newline=False)
    # Fetch the list of inverse registration transforms to map back to the native geometry.
    inv_transforms = registration['invtransforms']
    
    # Transform the binary whole tumor mask from atlas space back to the native T1c space.
    wt_mask_native_img = transform_to_native(native_img, wt_mask_atlas_img, inv_transforms)
    ants.image_write(wt_mask_native_img, str(scan_out_dir / f"{scan_name}_whole_tumor_mask.nii.gz"))
    
    # Transform the multi-class inner tumor segmentation from atlas space back to native space.
    it_native_img = transform_to_native(native_img, it_atlas_img, inv_transforms)
    ants.image_write(it_native_img, str(scan_out_dir / f"{scan_name}_inner_tumor_seg.nii.gz"))
    
    # Extract and save individual binary sub-masks (necrosis, enhancing, etc.) in native space.
    extract_sub_masks(it_native_img, scan_out_dir, scan_name)
    log("done.", timestamp=False)
        
    if cleanup:
        log(f"[{scan_name}] 5. Cleaning up intermediate files... ", newline=False)
        # Delete temporary atlas-aligned scans and intermediate mask files.
        paths_to_remove = [
            t1c_in_atlas_path,
            wt_raw_atlas_path,
            scan_out_dir / f"{scan_name}_whole_tumor_mask_SRI24.nii.gz",
            t1c_roi_path,
            it_raw_atlas_path
        ]
        for p in paths_to_remove:
            if p.exists(): p.unlink()
            
        # Delete the forward and inverse transform files generated by ANTs registration to save disk space.
        for t_list in [registration.get('fwdtransforms', []), registration.get('invtransforms', [])]:
            for t_path in t_list:
                p = Path(t_path)
                if p.exists(): p.unlink()
        log("done.", timestamp=False)

    log(f"[{scan_name}] Processing complete! Outputs saved in {scan_out_dir}\n")

def main():
    """
    Parses command line arguments and runs process_scan.
    """
    parser = argparse.ArgumentParser(description="Run TumorSynth pipeline on a single native T1c NIfTI scan.")
    parser.add_argument("input_t1c", type=Path, help="Path to the input native T1c NIfTI scan")
    parser.add_argument("-o", "--output-dir", type=Path, default=None, 
                        help="Directory to save the outputs. Defaults to the same directory as the input scan.")
    parser.add_argument("-a", "--atlas-dir", type=Path, default=None,
                        help="Path to the SRI24 atlas directory. Defaults to assets/SRI24_atlas relative to this script.")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run TumorSynth using GPU")
    parser.add_argument("-c", "--cleanup", action="store_true", help="Delete intermediate files, keeping only final native-space segmentations")
    args = parser.parse_args()

    # Check for FSL dependencies upfront to avoid failing after the long registration step.
    import shutil
    missing_deps = []
    for dep in ('fslmaths', 'fslmerge'):
        if not shutil.which(dep):
            missing_deps.append(dep)
    if missing_deps:
        log("Error: Missing required FSL (FMRIB Software Library) dependencies: " + ", ".join(missing_deps))
        log("Please ensure FSL is installed and available in your PATH.")
        log("On a cluster, you may need to run 'module load fsl' or source your FSL configuration.")
        return

    # Resolve the input path and verify it exists before processing.
    input_path = args.input_t1c.resolve()
    if not input_path.exists():
        log(f"Error: Input file {input_path} does not exist.")
        return

    # Use the user-defined output directory or fall back to the input file's parent directory.
    out_dir = args.output_dir.resolve() if args.output_dir else input_path.parent
    
    # Locate the atlas templates folder, defaulting to the assets/SRI24_atlas directory in this repo.
    base_dir = Path(__file__).parent.absolute()
    atlas_dir = args.atlas_dir.resolve() if args.atlas_dir else base_dir / 'assets' / 'SRI24_atlas'

    if not atlas_dir.exists():
        log(f"Error: Atlas directory {atlas_dir} does not exist.")
        return

    # Run the pipeline on the specified scan.
    process_scan(input_path, out_dir, atlas_dir, use_gpu=args.gpu, cleanup=args.cleanup)

if __name__ == "__main__":
    main()
