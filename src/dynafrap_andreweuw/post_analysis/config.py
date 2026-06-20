# =============================================================================
# Configuration
# =============================================================================

from pathlib import Path

# -----------------------------------------------------------------------------
# Input / output paths
# -----------------------------------------------------------------------------

# Folder containing the per-sample measurement CSV files.
# Each CSV should contain at least:
#   - time/frame column
#   - target ROI intensity column
#   - control ROI intensity column
INPUT_FOLDER = Path("data/output/measurements/pavement_lr_hr")

# Output CSV file where fitted FRAP parameters are saved.
FIT_RESULTS_CSV = Path("data/output/fit_results.csv")

# If True, save the fit results dataframe to FIT_RESULTS_CSV.
SAVE_FIT_RESULTS = True


# -----------------------------------------------------------------------------
# Experimental groups
# -----------------------------------------------------------------------------

# Group names used to assign files to experimental conditions.
# A file is assigned to a group if the group name appears in the filename.
GROUP_NAMES = [
    "Low-red",
    "High-Red",
]

# Colors used when plotting each group.
# These are matplotlib color names.
GROUP_COLORS = {
    "Low-red": "tab:blue",
    "High-Red": "tab:red",
}


# -----------------------------------------------------------------------------
# Input dataframe column names
# -----------------------------------------------------------------------------

# Column containing the time value or frame index.
# In the current analysis this represents frame number.
TIME_COL = "time"

# Column containing the raw intensity of the bleached target ROI.
TARGET_COL = "target_mean"

# Column containing the raw intensity of the unbleached control/reference ROI.
# This is used to correct acquisition-related photobleaching or drift.
CONTROL_COL = "control_mean"


# -----------------------------------------------------------------------------
# Metadata columns added during processing
# -----------------------------------------------------------------------------

# Column storing the experimental group assigned to each row.
GROUP_COL = "group"

# Column storing the source CSV filename for each row.
# This is used to count independent samples/files per group.
SOURCE_FILE_COL = "source_file"


# -----------------------------------------------------------------------------
# Output columns generated during postprocessing
# -----------------------------------------------------------------------------

# Per-frame control bleaching factor:
#
#   B(t) = C(t) / C0
#
# where:
#   C(t) = control intensity at frame t
#   C0   = control intensity at the pre-bleach frame
CONTROL_BLEACH_FACTOR_COL = "control_bleach_factor"

# Corrected pre-bleach target intensity:
#
#   I_pre = Tcorr(0)
#
# This is the target intensity after control-based photobleaching correction.
I_PRE_COL = "I_pre"

# Final normalized recovery column:
#
#   CorrectedRecovery%(t)
#       = (Tcorr(t) - I_post) / (I_pre - I_post) * 100
#
# This makes:
#   pre-bleach frame              = 100%
#   immediate post-bleach frame   = 0%
CORRECTED_TARGET_RECOVERY_PERCENT_COL = "target_corrected_recovery_percent"


# -----------------------------------------------------------------------------
# Frame selection
# -----------------------------------------------------------------------------

# Frame/time corresponding to the pre-bleach reference.
# This frame is used to define:
#   C0    = control pre-bleach intensity
#   I_pre = corrected target pre-bleach intensity
PREBLEACH_TIME = 0

# Minimum frame/time retained for plotting and fitting.
# With MIN_PLOT_TIME = 1, the pre-bleach frame is excluded from the plotted
# recovery curve, so the curve starts at the immediate post-bleach frame.
MIN_PLOT_TIME = 1


# -----------------------------------------------------------------------------
# Confidence interval / percentile display
# -----------------------------------------------------------------------------

# Multiplier for approximate 95% confidence intervals:
#
#   CI = mean ± 1.96 * SEM
#
# Only used if SHOW_CI = True and the plotting code uses confidence intervals.
CI_MULTIPLIER = 1.96


# -----------------------------------------------------------------------------
# Exponential fit parameters
# -----------------------------------------------------------------------------

# Initial guess for the recovery rate constant k in:
#
#   I(t) = I_inf - A * exp(-k * t)
INITIAL_K = 0.1

# Lower bound for k during curve fitting.
# This prevents k from becoming zero or negative.
MIN_K = 1e-9

# Maximum number of function evaluations allowed during nonlinear fitting.
MAX_FIT_EVALUATIONS = 10000


# -----------------------------------------------------------------------------
# Figure layout
# -----------------------------------------------------------------------------

# Matplotlib figure size in inches: (width, height).
FIGSIZE = (10, 6)


# -----------------------------------------------------------------------------
# Plot visibility options
# -----------------------------------------------------------------------------

# If True, draw every individual sample trace.
# If False, only group-level summaries are shown.
SHOW_INDIVIDUAL_TRACES = False

# If True, draw confidence interval or percentile region/outline,
# depending on the plotting implementation.
SHOW_CI = True

# If True, show a marker or annotation for the fitted recovery half-time.
SHOW_HALFTIME_MARKER = True


# -----------------------------------------------------------------------------
# Individual trace styling
# -----------------------------------------------------------------------------

# Transparency of individual traces.
# Lower values make traces less visually dominant.
INDIVIDUAL_LINE_ALPHA = 0.20

# Line width of individual traces.
INDIVIDUAL_LINE_WIDTH = 1.0


# -----------------------------------------------------------------------------
# Group mean and fit styling
# -----------------------------------------------------------------------------

# Line width of the group mean recovery curve.
MEAN_LINE_WIDTH = 2.5

# Line width of the fitted exponential recovery curve.
FIT_LINE_WIDTH = 2.5


# -----------------------------------------------------------------------------
# Plot labels
# -----------------------------------------------------------------------------

# Main plot title.
TITLE = "FRAP recovery by group"

# Y-axis label.
# The plotted value is corrected recovery percentage:
#   0%   = immediate post-bleach intensity
#   100% = corrected pre-bleach intensity
Y_LABEL = "Corrected recovery [%]"

# X-axis label.
# The time column currently represents frame number, not seconds.
X_LABEL = "Time [# frames]"
