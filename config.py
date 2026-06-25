from pathlib import Path

# Path to the local TumorSynth tools folder in this repository
TUMORSYNTH_DIR = Path(__file__).parent.absolute() / 'tools' / 'tumorsynth'

# Set to True if you want to use GPU for segmentation, False to run on CPU.
USE_GPU = False

# Inner tumor sub-structure labels as output by TumorSynth
INNER_TUMOR_LABELS = {
    'necrosis': 1,         # Necrotic / Non-enhancing tumor core (NCR/NET)
    'enhancing_tumor': 4,  # Enhancing tumor (ET)
    'edema': 2,            # Peritumoral edema (ED)
}
