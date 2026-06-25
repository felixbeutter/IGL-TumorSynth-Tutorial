from pathlib import Path

# Path to the local TumorSynth tools folder in this repository
TUMORSYNTH_DIR = Path(__file__).parent.absolute() / 'tools' / 'tumorsynth'

# Set to True if you want to use GPU for segmentation, False to run on CPU.
USE_GPU = False

# BraTS-compliant inner tumor sub-structure labels as output by TumorSynth
INNER_TUMOR_LABELS = {
    'non_enhancing': 1,   # NET/Edema: Largest region, borders background
    'necrosis': 2,        # NCR: Necrotic core, surrounded by enhancing ring
    'enhancing': 3,       # ET: Gadolinium-enhancing ring
}
