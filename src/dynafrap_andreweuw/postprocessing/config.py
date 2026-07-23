# =============================================================================
# Configuration
# =============================================================================
#
# Quick start for a new experiment:
#   1. Set INPUT_FOLDERS (in paths.py) to the folder(s) that hold your
#      measurement CSVs.
#   2. Update GROUPS so each name/pattern/color matches one of your
#      experimental conditions.
#   3. Update the plot labels/title if needed.
#
# Everything under "Advanced / internal settings" further down rarely needs
# to change and is mostly here for reference.

import re
from dataclasses import dataclass

from dynafrap_andreweuw.paths import INPUT_FOLDERS, OUTPUT_DIR

# -----------------------------------------------------------------------------
# 1. Which data to analyze
# -----------------------------------------------------------------------------



# If True, save the fit results dataframe to fit_results_csv().
SAVE_FIT_RESULTS = True


def _slugify(text):
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")


def fit_results_csv():
    """
    Output CSV path for the fitted FRAP parameters.

    The filename encodes both the input folders (via INPUT_FOLDERS) and
    the current group combination (via GROUPS) so different runs don't
    overwrite each other's results.
    """
    folders_slug = "_".join(_slugify(folder.name) for folder in INPUT_FOLDERS)
    groups_slug = "_".join(_slugify(group.name) for group in GROUPS)
    return OUTPUT_DIR / f"fit_results_{folders_slug}_{groups_slug}.csv"


# -----------------------------------------------------------------------------
# 2. Experimental groups
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Group:
    # Descriptive label used in plots, legends, and fit results.
    name: str

    # Case-insensitive regex used to assign files to this group.
    # A file is assigned to a group if the pattern matches its filename.
    # The pattern is searched for inside of each folder in INPUT_FOLDERS.
    pattern: str

    # Matplotlib color used when plotting this group.
    color: str


# One entry per experimental condition. Add/remove entries to change the
# number of groups; name/pattern/color are independent so the display name
# can be descriptive without constraining the filename-matching regex.
GROUPS = [
    Group(name="Low-red", pattern=r"low-red", color="#0072B2"),
    Group(name="High-Red", pattern=r"high-red", color="#D55E00"),
    Group(name="WL-guard", pattern=r"wl-guard", color="#009E73"),
    Group(name="WL-pavement", pattern=r"wl-pavement", color="#E69F00"),
    Group(name="WL-palisade", pattern=r"wl-palisade", color="#CC79A7"),
    Group(name="WL-spongy", pattern=r"wl-spongy", color="#56B4E9"),
]


# -----------------------------------------------------------------------------
# 3. Frame selection (depends on your acquisition protocol)
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

# Maximum frame/time retained for postprocessing, plotting, and fitting.
# Frames with TIME_COL > MAX_PLOT_TIME are dropped before any further
# processing. Useful for quickly testing on a truncated recovery curve.
# Set to None to keep all frames.
MAX_PLOT_TIME = 20


# -----------------------------------------------------------------------------
# 4. Outlier handling
# -----------------------------------------------------------------------------

# If a frame's control_bleach_factor drops below this value, the control
# channel has essentially collapsed to noise, so
#   target_corrected = target / control_bleach_factor
# divides by a near-zero number and explodes to unreliable values.
# Frames below this threshold are flagged as outliers.
CONTROL_BLEACH_FACTOR_OUTLIER_THRESHOLD = 0.0

# If True, any file containing outlier frames is inpainted: its outlier
# frames are replaced with the group's per-timepoint mean (computed from the
# remaining, non-outlier frames) and the file is kept in the group
# mean/CI/fit.
#
# If False, any file containing outlier frames is excluded entirely from the
# group mean/CI/fit. Its full raw trace can still be drawn separately
# instead of silently disappearing (see PLOT_OUTLIERS).
# ... Could be also truncated since its usually the last frames.
CORRECT_OUTLIER = True

# If True (and CORRECT_OUTLIER is False), draw the full raw trace of each
# excluded file, clearly labeled, so the excluded data stays visible.
PLOT_OUTLIERS = True


# -----------------------------------------------------------------------------
# 5. Plot labels
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


# -----------------------------------------------------------------------------
# 6. Plot content: what to show
# -----------------------------------------------------------------------------

# If True, draw every individual sample trace.
# If False, only group-level summaries are shown.
SHOW_INDIVIDUAL_TRACES = False

# If True, draw confidence interval or percentile region/outline,
# depending on the plotting implementation.
SHOW_CI = True

# If True, show a marker or annotation for the fitted recovery half-time.
SHOW_HALFTIME_MARKER = True

# Multiplier for approximate 95% confidence intervals:
#
#   CI = mean ± 1.96 * SEM
#
# Only used if SHOW_CI = True and the plotting code uses confidence intervals.
CI_MULTIPLIER = 1.96


# =============================================================================
# Advanced / internal settings (rarely need to change)
# =============================================================================

# -----------------------------------------------------------------------------
# Input dataframe column names
# -----------------------------------------------------------------------------
# Only change these if your CSV files use different column names.

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

# Column storing whether a frame's photobleaching correction is considered
# unreliable (see CONTROL_BLEACH_FACTOR_OUTLIER_THRESHOLD above).
IS_OUTLIER_COL = "is_outlier"


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

# Matplotlib maximum number of ticks so that the figure is sensible
X_AXIS_TICK_LIMIT = 10000

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
