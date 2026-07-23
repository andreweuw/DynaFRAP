from pathlib import Path

# 1. Preprocessing
# uv run dynafrap-preprocess

INPUT_LIF = Path("data/lif/Guard-1.5hDark-5min-High-Red-Test.lif")
OUTPUT_TIFF_DIRECTORY = Path("data/tif")

# 2. Segmentation
# uv run dynafrap-segment

UNET_LOG_DIR = "data/checkpoints/log/20240314-211347-unet_default_fold0"
# Prediction input directory:
PRED_DIR_IN = "data/tif/Guard-1.5hDark-5min-High-Red-Test"
# Prediction output directory:
PRED_DIR_OUT_BASE = "data/segmentations/Guard-1.5hDark-5min-High-Red-Test"

# 3. Measuring
# uv run dynafrap-measure

SEGMENTATIONS_FOLDER = Path("data/segmentations")
MEASUREMENTS_INPUT_FOLDER = Path("data/tif")
MEASUREMENTS_OUTPUT_FOLDER = Path("data/measurements")

# 4. Postprocessing
# uv run dynafrap-postprocess


# Full paths to the folders to process. CSV files from all of these folders
# are pooled together into a single combined run (one plot, one fit results
# CSV). Each folder holds the per-sample measurement CSV files for that dataset.
# Each CSV should contain at least:
#   - time/frame column
#   - target ROI intensity column
#   - control ROI intensity column
INPUT_FOLDERS = [
    Path("data/measurements/wl"),
    Path("data/measurements/palisade_redo"),
]

OUTPUT_DIR = Path("data/fits")
