# TumorSynth Tutorial

This repository provides a simple tutorial on how to use **TumorSynth** for brain tumor segmentation. It demonstrates how to take native multi-modal MRI scans (e.g., T1c, T2, FLAIR), align them to a common atlas (SRI24), run TumorSynth to segment the whole tumor and inner tumor sub-structures, and transform the segmentations back to the original native space.

---

## Prerequisites & Installation

Before running this tutorial, you need to set up FSL and the required Python environment on your machine.

### 1. Install FSL (FMRIB Software Library)
The internal [mri_TumorSynth](tools/tumorsynth/mri_TumorSynth) wrapper script requires FSL tools (specifically `fslmaths` and `fslmerge`) to combine and process segmentations.
- If you do not have FSL installed, download and install it from the [FSL Installation Guide](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation).
- Make sure FSL is in your system's `PATH`. You can verify this by running:
  ```bash
  which fslmaths
  ```

### 2. Set Up the Python Conda Environment
Because the trained model weights are very large (~3.5GB), they are ignored by git and are not included in a fresh clone. You must download them first.

1. **Download the pre-trained model weights:**
   - Follow the instructions on the [official FreeSurfer Wiki](https://surfer.nmr.mgh.harvard.edu/fswiki/TumorSynth#Installation) to download the model zip files for the whole tumor (`TumorSynth_v1.0.zip`) and the inner tumor (`Task003_InnerTumor.zip`).
   - Extract the zip files so that the directories `Task002_Tumor` and `Task003_InnerTumor` are located under `tools/tumorsynth/models/`. After extraction, the directory structure must be:
     - `tools/tumorsynth/models/Task002_Tumor/nnUNetTrainerV2__nnUNetPlansv2.1/plans.pkl`
     - `tools/tumorsynth/models/Task003_InnerTumor/nnUNetTrainerV2__nnUNetPlansv2.1/plans.pkl`

2. **Run the Environment Setup Script:**
   Use the provided bash script [create_nnUNet_v1.7_env.sh](tools/tumorsynth/create_nnUNet_v1.7_env.sh) to create a conda environment (named `nnUNet_v1.7`) and install all required machine learning and processing dependencies.
   
   If `conda` is active in your terminal, the script will automatically locate your `conda.sh` profile script.
   
   On **macOS** (CPU-only, no CUDA support):
   ```bash
   cd tools/tumorsynth
   chmod +x create_nnUNet_v1.7_env.sh
   ./create_nnUNet_v1.7_env.sh -n nnUNet_v1.7 -m models -d .
   cd ../..
   ```
   
   On **Linux** (with GPU / CUDA acceleration support):
   ```bash
   cd tools/tumorsynth
   chmod +x create_nnUNet_v1.7_env.sh
   ./create_nnUNet_v1.7_env.sh -n nnUNet_v1.7 -m models -d . -c
   cd ../..
   ```
   
   *Notes:*
   - *GPU Support:* The `-c` flag enables GPU acceleration by installing CUDA support. Do not pass the `-c` flag on a Mac, as CUDA is not available on macOS.
   - *Conda Auto-Detection:* If the script cannot auto-detect conda, you can manually specify the path using the `-e` flag (e.g., `./create_nnUNet_v1.7_env.sh -e "$(conda info --base)/etc/profile.d/conda.sh" -n nnUNet_v1.7 -m models -d .` or pointing directly to your conda profile script).

3. **Install ANTsPy:**
   ANTsPy is used for registering patient scans to the SRI24 atlas. Activate the new conda environment and install `antspyx`:
   ```bash
   conda activate nnUNet_v1.7
   pip install antspyx
   ```

### 3. Source the Environment Variables
nnU-Net requires environment variables to locate pre-trained weights. A helper script named [nnUNet_v1.7_path.sh](tools/tumorsynth/nnUNet_v1.7_path.sh) is generated in `tools/tumorsynth/` during installation.

Before running the pipeline, source this path file:
```bash
source tools/tumorsynth/nnUNet_v1.7_path.sh
```

> [!TIP]
> To load these paths automatically whenever the conda environment is activated, activate the environment first and then copy the generated script to the conda activation folder:
> ```bash
> conda activate nnUNet_v1.7
> mkdir -p $CONDA_PREFIX/etc/conda/activate.d
> cp tools/tumorsynth/nnUNet_v1.7_path.sh $CONDA_PREFIX/etc/conda/activate.d/
> ```

---

## Running a Scan

The `assets/` folder is structured as follows:
- `SRI24_atlas/`: Contains the reference atlas and brain mask required for registration.

To run the pipeline on a scan:

1. Ensure the conda environment is active and environment variables are sourced:
   ```bash
   conda activate nnUNet_v1.7
   source tools/tumorsynth/nnUNet_v1.7_path.sh
   ```

2. Run the main pipeline script [run_tumorsynth.py](run_tumorsynth.py) by providing the path to your NIfTI scans:

   On **macOS** (CPU-only):
   ```bash
   python run_tumorsynth.py -i path_to_scan.nii.gz -o outputs -c
   ```

   On **Linux/HPC** (with GPU):
   ```bash
   python run_tumorsynth.py -i path_to_scan.nii.gz -o outputs -c --gpu
   ```

### Command Line Arguments

- `-i`, `--inputs`: (Required) Path to one or more `.nii.gz` scans (e.g., T1c, T2, FLAIR). If multiple scans are provided, they **must be co-registered** to each other in their native space.
- `--reg-input`: (Optional) Path to the specific input file used for atlas registration. Defaults to the first input. Using a T1 or T1c scan is recommended.
- `--scan-name`: (Optional) Base name for the outputs. Defaults to the name of the `--reg-input` without its extension.
- `-o`, `--output-dir`: (Optional) Path to save the outputs. Defaults to the same directory as the reg-input scan.
- `-a`, `--atlas-dir`: (Optional) Path to the SRI24 atlas. Defaults to `assets/SRI24_atlas/`.
- `-g`, `--gpu`: (Optional) Flag to run TumorSynth with GPU acceleration.
- `-c`, `--cleanup`: (Optional) Flag to automatically delete intermediate atlas-space files and transforms, keeping only the final native space segmentations.

### Running on a SLURM Cluster

If you are running the pipeline on an HPC cluster with SLURM, a submission script [slurm_wrapper.sh](slurm_wrapper.sh) is provided at the root of the repository.

To submit the segmentation job, run:
```bash
sbatch slurm_wrapper.sh -i assets/sample_scans/UCSF-PDGM-0004_T1c.nii.gz --gpu --cleanup -o outputs
```

This script automatically activates the `nnUNet_v1.7` conda environment, sets up the dynamic path variables, and submits the job to the GPU partition. Logs will be written to `logs/tumorsynth_<JOBID>.out`.

---

## What the Script Does

For the given scan, the script will:

1. **Register to Atlas**: Perform a non-linear (SyN) registration of your registration modality (e.g., T1c) to the SRI24 T1 atlas using `antspyx`, and map all other input modalities into this atlas space.
2. **Whole Tumor Segmentation**: Run `mri_TumorSynth --wholetumor` to segment the full brain and tumor (defined in [segmentation.py](segmentation.py)).
3. **Inner Tumor Segmentation**: Create a masked ROI of the tumor and run `mri_TumorSynth --innertumor` to identify specific sub-structures (necrosis, enhancing tumor, and peritumoral edema).
4. **Native Space Transformation**: Map all resulting masks and segmentations back to the original patient's native space and save them in the `outputs/` directory.

### Outputs Structure

All results are stored in a scan-specific directory under the chosen output directory (e.g., `outputs/UCSF-PDGM-0004_T1c/`):
- `<scan_name>_whole_tumor_mask.nii.gz`: Binary mask of the whole tumor in native space.
- `<scan_name>_brain_tissue_seg.nii.gz`: Complete 18-label parcellation of healthy brain tissues (labels 1-17) and the whole tumor (label 18) in native space.
- `<scan_name>_inner_tumor_seg.nii.gz`: Multi-label segmentation of inner tumor structures in native space.
- `<scan_name>_necrosis_mask.nii.gz`: Binary mask of the necrotic / non-enhancing tumor core (Label 1, NCR/NET).
- `<scan_name>_enhancing_tumor_mask.nii.gz`: Binary mask of the enhancing tumor (Label 2, ET).
- `<scan_name>_edema_mask.nii.gz`: Binary mask of the peritumoral edema (Label 3, ED).

These structures are configured in [config.py](config.py).
