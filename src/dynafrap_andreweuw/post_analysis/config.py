# =============================================================================
# Configuration
# =============================================================================

from pathlib import Path

INPUT_FOLDER = Path("data/output/measurements/pavement_lr_hr")

GROUP_NAMES = [
    "Low-red",
    "High-Red",
]

GROUP_COLORS = {
    "Low-red": "tab:blue",
    "High-Red": "tab:red",
}

TIME_COL = "time"
TARGET_COL = "target_mean"
CONTROL_COL = "control_mean"

GROUP_COL = "group"
SOURCE_FILE_COL = "source_file"

CONTROL_BLEACH_FACTOR_COL = "control_bleach_factor"
CORRECTED_TARGET_COL = "target_mean_corrected"
I_PRE_COL = "I_pre"
CORRECTED_TARGET_PERCENT_COL = "target_mean_corrected_percent"

PREBLEACH_TIME = 0
MIN_PLOT_TIME = 1

# Options:
#
#   "raw_corrected"
#       Fit/plot raw photobleaching-corrected intensities.
#
#   "percent_of_own_prebleach"
#       Fit/plot each sample as percent of its own corrected pre-bleach intensity.
#
# NORMALIZATION_MODE = "percent_of_own_prebleach"
NORMALIZATION_MODE = "raw_corrected"

CI_MULTIPLIER = 1.96

INITIAL_K = 0.1
MIN_K = 1e-9
MAX_FIT_EVALUATIONS = 10000

FIGSIZE = (10, 6)

SHOW_INDIVIDUAL_TRACES = True
SHOW_CI = True
SHOW_HALFTIME_MARKER = True

INDIVIDUAL_LINE_ALPHA = 0.20
INDIVIDUAL_LINE_WIDTH = 1.0

MEAN_LINE_WIDTH = 2.5
FIT_LINE_WIDTH = 2.5

SAVE_FIT_RESULTS = True
FIT_RESULTS_CSV = Path("fit_results.csv")
