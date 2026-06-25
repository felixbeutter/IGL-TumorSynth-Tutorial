# TumorSynth Tutorial

This repository provides a standalone, clean, and simple tutorial on how to use **TumorSynth** for brain tumor segmentation. It demonstrates how to take native T1c MRI scans, align them to a common atlas (SRI24), run TumorSynth to segment the whole tumor and inner tumor sub-structures, and transform the segmentations back to the original native space.

---

## Prerequisites & Installation

Before running this tutorial, you need to set up FSL and the required Python environment on your machine.

### 1. Install FSL (FMRIB Software Library)
The internal [mri_TumorSynth](file:///Users/felix/GitHub/IGL-TumorSynth-Tutorial/tools/tumorsynth/mri_TumorSynth) wrapper script requires FSL tools (specifically `fslmaths` and `fslmerge`) to combine and process segmentations.
- If you do not have FSL installed, download and install it from the [FSL Installation Guide](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation).
- Make sure FSL is in your system's `PATH`. You can verify this by running:
  ```bash
  which fslmaths
  ```

### 2. Set Up the Python Conda Environment
Because the trained model weights are very large (~3.5GB), they are ignored by git and are not included in a fresh clone. You must download them first.

1. **Download the pre-trained model weights:**
   - Follow the instructions on the [official FreeSurfer Wiki](https://surfer.nmr.mgh.harvard.edu/fswiki/TumorSynth#Installation) to download the model zip files for the whole tumor (`TumorSynth_v1.0.zip`) and the inner tumor (`Task003_InnerTumor.zip`).
   - Extract the contents so that they are located precisely under `tools/tumorsynth/models/Task002_Tumor/` and `tools/tumorsynth/models/Task003_InnerTumor/`.

2. **Run the Environment Setup Script:**
   Use the provided bash script [create_nnUNet_v1.7_env.sh](file:///Users/felix/GitHub/IGL-TumorSynth-Tutorial/tools/tumorsynth/create_nnUNet_v1.7_env.sh) to create a conda environment (named `nnUNet_v1.7`) and install all required machine learning and processing dependencies.
   
   If `conda` is active in your terminal, the script will automatically locate your `conda.sh` profile script. Simply run:
   ```bash
   cd tools/tumorsynth
   chmod +x create_nnUNet_v1.7_env.sh
   ./create_nnUNet_v1.7_env.sh -n nnUNet_v1.7 -m models -d .
   cd ../..
   ```
   
   *Note: If the script cannot auto-detect conda, you can manually specify the path using the `-e` flag (e.g., `./create_nnUNet_v1.7_env.sh -e "$(conda info --base)/etc/profile.d/conda.sh" -n nnUNet_v1.7 -m models -d .` or pointing directly to your conda profile script).*

3. **Install ANTsPy:**
   ANTsPy is used for registering patient scans to the SRI24 atlas. Activate the new conda environment and install `antspyx`:
   ```bash
   conda activate nnUNet_v1.7
   pip install antspyx
   ```

### 3. Source the Environment Variables
nnU-Net requires environment variables to locate pre-trained weights. A helper script named [nnUNet_v1.7_path.sh](file:///Users/felix/GitHub/IGL-TumorSynth-Tutorial/tools/tumorsynth/nnUNet_v1.7_path.sh) is generated in `tools/tumorsynth/` during installation.

Before running the pipeline, source this path file:
```bash
source tools/tumorsynth/nnUNet_v1.7_path.sh
```

> [!TIP]
> To load these paths automatically whenever the conda environment is activated:
> ```bash
> mkdir -p $CONDA_PREFIX/etc/conda/activate.d
> cp tools/tumorsynth/nnUNet_v1.7_path.sh $CONDA_PREFIX/etc/conda/activate.d/
> ```

---

## Running a Scan

The `assets/` folder is structured as follows:
- `SRI24_atlas/`: Contains the reference atlas and brain mask required for registration.
- `sample_scans/`: Place your raw, native space `.nii.gz` T1c scans here (includes UCSF sample scans).

To run the pipeline on a scan:

1. Ensure the conda environment is active and environment variables are sourced:
   ```bash
   conda activate nnUNet_v1.7
   source tools/tumorsynth/nnUNet_v1.7_path.sh
   ```

2. Run the main pipeline script [run_tumorsynth.py](file:///Users/felix/GitHub/IGL-TumorSynth-Tutorial/run_tumorsynth.py) by providing the path to a T1c NIfTI scan:
   ```bash
   python run_tumorsynth.py assets/sample_scans/UCSF-PDGM-0004_T1c.nii.gz -o outputs -c
   ```

### Command Line Arguments

- `input_t1c`: (Required) Path to the native T1c `.nii.gz` scan.
- `-o`, `--output-dir`: (Optional) Path to save the outputs. Defaults to the same directory as the input scan.
- `-a`, `--atlas-dir`: (Optional) Path to the SRI24 atlas. Defaults to `assets/SRI24_atlas/`.
- `-g`, `--gpu`: (Optional) Flag to run TumorSynth with GPU acceleration.
- `-c`, `--cleanup`: (Optional) Flag to automatically delete intermediate atlas-space files and transforms, keeping only the final native space segmentations.

---

## What the Script Does

For the given scan, the script will:

1. **Register to Atlas**: Perform a non-linear (SyN) registration of your native T1c scan to the SRI24 T1 atlas using `antspyx` (defined in [registration.py](file:///Users/felix/GitHub/IGL-TumorSynth-Tutorial/registration.py)).
2. **Whole Tumor Segmentation**: Run `mri_TumorSynth --wholetumor` to segment the full brain and tumor (defined in [segmentation.py](file:///Users/felix/GitHub/IGL-TumorSynth-Tutorial/segmentation.py)).
3. **Inner Tumor Segmentation**: Create a masked ROI of the tumor and run `mri_TumorSynth --innertumor` to identify specific sub-structures (non-enhancing, necrosis, and enhancing ring).
4. **Native Space Transformation**: Map all resulting masks and segmentations back to the original patient's native space and save them in the `outputs/` directory.

### Inner Tumor Sub-structures
The inner tumor model outputs the following BraTS-compliant labels in native space (configured in [config.py](file:///Users/felix/GitHub/IGL-TumorSynth-Tutorial/config.py)):
- **Label 1 (non_enhancing)**: NET/Edema (Largest region, borders background). Saved as `*_non_enhancing_mask.nii.gz`.
- **Label 2 (necrosis)**: NCR (Necrotic core). Saved as `*_necrosis_mask.nii.gz`.
- **Label 3 (enhancing)**: ET (Gadolinium-enhancing ring). Saved as `*_enhancing_mask.nii.gz`.
