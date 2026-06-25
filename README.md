# TumorSynth Tutorial

This repository provides a standalone, clean, and simple tutorial on how to use **TumorSynth** for brain tumor segmentation. It demonstrates how to take native T1c MRI scans, align them to a common atlas (SRI24), run TumorSynth to segment the whole tumor and inner tumor sub-structures, and transform the segmentations back to the original native space.

## Prerequisites

Before running this tutorial, you need to set up TumorSynth and the required Python environment on your machine.

### 1. Setting up TumorSynth

TumorSynth relies on `nnU-Net` and requires specific model weights. Follow these steps to install it on a clean machine:

1. **Clone the TumorSynth Repository:**
   ```bash
   git clone https://github.com/fprados/TumorSynth.git
   cd TumorSynth
   ```

2. **Create a Conda Environment for TumorSynth:**
   It is highly recommended to run TumorSynth within its own isolated Conda environment.
   ```bash
   conda create -n nnUNet_v1.7 python=3.9 -y
   conda activate nnUNet_v1.7
   ```

3. **Install Dependencies:**
   Follow the installation instructions in the TumorSynth repository. Typically, this involves installing `nnunet`:
   ```bash
   pip install nnunet
   ```
   *Note: Refer to the [official FreeSurfer Wiki](https://surfer.nmr.mgh.harvard.edu/fswiki/TumorSynth) or the [GitHub repo](https://github.com/fprados/TumorSynth) for exact dependency requirements, as you will also need to download the pre-trained weights (`TumorSynth_v1.0.zip` and `Task003_InnerTumor.zip`) and set the appropriate environment variables (`RESULTS_FOLDER`, etc.).*

4. **Verify Installation:**
   Ensure the `mri_TumorSynth` command is available in your terminal:
   ```bash
   mri_TumorSynth --help
   ```

### 2. Setting up the Tutorial Environment

The tutorial script `tutorial.py` requires `antspyx` for image registration (aligning your scans to the SRI24 atlas) and basic image manipulations.

With your Conda environment activated (`conda activate nnUNet_v1.7`), install `antspyx`:

```bash
pip install antspyx
```

## Running the Tutorial

The `assets/` folder is structured as follows:
- `SRI24_atlas/`: Contains the reference atlas and brain mask required for registration.
- `sample_scans/`: Place your raw, native space `.nii.gz` T1c scans here. *(Note: This folder is ignored by git to prevent uploading large files).*

To run the pipeline on all scans in `sample_scans`:

1. Ensure your environment is activated:
   ```bash
   conda activate nnUNet_v1.7
   ```

2. Run the script:
   ```bash
   python tutorial.py
   ```


## What the Script Does

For each scan found in `assets/sample_scans/`, the script will:

1. **Register to Atlas**: Perform a non-linear (SyN) registration of your native T1c scan to the SRI24 T1 atlas using `antspyx`.
2. **Whole Tumor Segmentation**: Run `mri_TumorSynth --wholetumor` to segment the full brain and tumor.
3. **Inner Tumor Segmentation**: Create a masked ROI of the tumor and run `mri_TumorSynth --innertumor` to identify specific sub-structures (non-enhancing, necrosis, and enhancing ring).
4. **Native Space Transformation**: Map all resulting masks and segmentations back to the original patient's native space and save them in the `outputs/` directory.

### Inner Tumor Sub-structures
The inner tumor model outputs the following BraTS-compliant labels:
- **Label 1 (non_enhancing)**: NET/Edema (Largest region, borders background).
- **Label 2 (necrosis)**: NCR (Necrotic core).
- **Label 3 (enhancing)**: ET (Gadolinium-enhancing ring).
