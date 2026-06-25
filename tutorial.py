from pathlib import Path
import ants

from registration import register_to_atlas, transform_to_native
from segmentation import execute_tumorsynth, extract_whole_tumor_mask, extract_sub_masks

def process_scan(native_t1c_path: Path, output_dir: Path, atlas_dir: Path):
    """
    Processes a single T1c scan through the complete TumorSynth pipeline.
    """
    # Safe name splitting to handle .nii.gz and other dots in name
    scan_name = native_t1c_path.name.split('.nii')[0]
    scan_out_dir = output_dir / scan_name
    scan_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define atlas paths
    sri24_t1 = atlas_dir / 'SRI24_T1.nii.gz'
    sri24_mask = atlas_dir / 'SRI24_brain_mask.nii.gz'
    
    print(f"\n[{scan_name}] 1. Registering to SRI24 Atlas...")
    native_img = ants.image_read(str(native_t1c_path))
    atlas_img = ants.image_read(str(sri24_t1))
    atlas_mask = ants.image_read(str(sri24_mask))
    
    t1c_in_atlas_img, registration = register_to_atlas(native_img, atlas_img, atlas_mask)
    t1c_in_atlas_path = scan_out_dir / f"{scan_name}_t1c_SRI24.nii.gz"
    ants.image_write(t1c_in_atlas_img, str(t1c_in_atlas_path))
    
    print(f"[{scan_name}] 2. Running Whole Tumor Segmentation...")
    wt_raw_atlas_path = scan_out_dir / "tumorsynth_wt_raw_SRI24.nii.gz"
    execute_tumorsynth(t1c_in_atlas_path, wt_raw_atlas_path, mask_type='wholetumor')
    
    wt_atlas_img = ants.image_read(str(wt_raw_atlas_path))
    wt_mask_atlas_img = extract_whole_tumor_mask(wt_atlas_img)
    ants.image_write(wt_mask_atlas_img, str(scan_out_dir / f"{scan_name}_whole_tumor_mask_SRI24.nii.gz"))
    
    print(f"[{scan_name}] 3. Running Inner Tumor Segmentation...")
    # Inner tumor model requires a masked tumor ROI as input
    t1c_roi_img = t1c_in_atlas_img * wt_mask_atlas_img
    t1c_roi_path = scan_out_dir / f"{scan_name}_t1c_roi_SRI24.nii.gz"
    ants.image_write(t1c_roi_img, str(t1c_roi_path))
    
    it_raw_atlas_path = scan_out_dir / "tumorsynth_it_raw_SRI24.nii.gz"
    execute_tumorsynth(t1c_roi_path, it_raw_atlas_path, mask_type='innertumor')
    it_atlas_img = ants.image_read(str(it_raw_atlas_path))
    
    print(f"[{scan_name}] 4. Transforming results back to Native Space...")
    inv_transforms = registration['invtransforms']
    
    wt_mask_native_img = transform_to_native(native_img, wt_mask_atlas_img, inv_transforms)
    ants.image_write(wt_mask_native_img, str(scan_out_dir / f"{scan_name}_whole_tumor_mask.nii.gz"))
    
    it_native_img = transform_to_native(native_img, it_atlas_img, inv_transforms)
    ants.image_write(it_native_img, str(scan_out_dir / f"{scan_name}_inner_tumor_seg.nii.gz"))
    
    # Save individual sub-masks directly
    extract_sub_masks(it_native_img, scan_out_dir, scan_name)
        
    print(f"[{scan_name}] Processing complete! Outputs saved in {scan_out_dir}")

def main():
    base_dir = Path(__file__).parent.absolute()
    assets_dir = base_dir / 'assets'
    sample_scans_dir = assets_dir / 'sample_scans'
    atlas_dir = assets_dir / 'SRI24_atlas'
    output_dir = base_dir / 'outputs'
    
    output_dir.mkdir(exist_ok=True)
    
    if not sample_scans_dir.exists():
        print(f"Error: Sample scans directory not found at {sample_scans_dir}")
        return
        
    scans = list(sample_scans_dir.glob('*.nii.gz'))
    if not scans:
        print("No .nii.gz files found in sample_scans directory.")
        return
        
    print(f"Found {len(scans)} scans to process.")
    for scan in scans:
        process_scan(scan, output_dir, atlas_dir)

if __name__ == "__main__":
    main()
