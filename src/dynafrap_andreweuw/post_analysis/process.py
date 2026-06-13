import numpy as np

from dynafrap_andreweuw.post_analysis.config import (
    CONTROL_BLEACH_FACTOR_COL,
    CONTROL_COL,
    CORRECTED_TARGET_COL,
    CORRECTED_TARGET_PERCENT_COL,
    I_PRE_COL,
    MIN_PLOT_TIME,
    NORMALIZATION_MODE,
    PREBLEACH_TIME,
    SOURCE_FILE_COL,
    TARGET_COL,
    TIME_COL,
)
from dynafrap_andreweuw.post_analysis.frap_math import fit_recovery_curve, mobile_fraction_from_fit

# =============================================================================
# Column / normalization helpers
# =============================================================================


def get_y_col_from_normalization_mode():
    if NORMALIZATION_MODE == "raw_corrected":
        return CORRECTED_TARGET_COL

    if NORMALIZATION_MODE == "percent_of_own_prebleach":
        return CORRECTED_TARGET_PERCENT_COL

    raise ValueError(
        "NORMALIZATION_MODE must be either 'raw_corrected' or 'percent_of_own_prebleach'"
    )


def get_y_label(y_col):
    if y_col == CORRECTED_TARGET_COL:
        return "Photobleaching-corrected target intensity"

    if y_col == CORRECTED_TARGET_PERCENT_COL:
        return "Corrected target intensity (% of own pre-bleach)"

    return y_col


def get_I_pre_for_fit(group_df):
    """
    Return I_pre in the same units as the selected normalization mode.

    For raw corrected intensities:
        I_pre = mean corrected pre-bleach intensity across samples

    For percent-normalized curves:
        I_pre = 100
    """
    if NORMALIZATION_MODE == "raw_corrected":
        return group_df[I_PRE_COL].mean()

    if NORMALIZATION_MODE == "percent_of_own_prebleach":
        return 100.0

    raise ValueError(f"Unknown NORMALIZATION_MODE: {NORMALIZATION_MODE}")


# =============================================================================
# Data processing functions
# =============================================================================


def correct_target(df):
    """
    Apply photobleaching correction to one sample dataframe.

    Steps:

    1. Find the pre-bleach frame.
    2. Estimate control bleaching factor:

           control_bleach_factor(t) = control(t) / control(prebleach)

    3. Correct target intensity:

           target_corrected(t) = target(t) / control_bleach_factor(t)

    4. Store I_pre:

           I_pre = target_corrected(prebleach)

    5. Create per-sample percentage normalization:

           target_corrected_percent(t) = target_corrected(t) / I_pre * 100

    6. Return only frames from MIN_PLOT_TIME onward.
    """
    df = df.copy()
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    prebleach_rows = df[df[TIME_COL] == PREBLEACH_TIME]

    if prebleach_rows.empty:
        raise ValueError(f"No pre-bleach frame found at {TIME_COL} == {PREBLEACH_TIME}")

    prebleach_idx = prebleach_rows.index[0]

    target = df[TARGET_COL]
    control = df[CONTROL_COL]

    control_prebleach = control.iloc[prebleach_idx]

    if control_prebleach == 0:
        raise ValueError(
            "Pre-bleach control intensity is zero; cannot perform photobleaching correction."
        )

    control_bleach_factor = control / control_prebleach
    target_corrected = target / control_bleach_factor

    I_pre = target_corrected.iloc[prebleach_idx]

    if I_pre == 0 or not np.isfinite(I_pre):
        target_corrected_percent = np.nan
    else:
        target_corrected_percent = (target_corrected / I_pre) * 100.0

    df[CONTROL_BLEACH_FACTOR_COL] = control_bleach_factor
    df[CORRECTED_TARGET_COL] = target_corrected
    df[I_PRE_COL] = I_pre
    df[CORRECTED_TARGET_PERCENT_COL] = target_corrected_percent

    return df[df[TIME_COL] >= MIN_PLOT_TIME].copy()


def fit_group_mean(summary, group_df, group_name, y_col):
    """
    Fit one-phase exponential recovery to the group mean curve.
    """
    fit_params, fit_curve = fit_recovery_curve(
        time_values=summary[TIME_COL],
        intensity_values=summary["mean"],
    )

    if fit_params is None:
        print(f"Fit failed or not enough points for group: {group_name}")
        return None, None

    I_pre = get_I_pre_for_fit(group_df)

    mobile_fraction, immobile_fraction = mobile_fraction_from_fit(
        I_pre=I_pre,
        I_0=fit_params["I_0"],
        A=fit_params["A"],
    )

    fit_result = {
        "group": group_name,
        "normalization_mode": NORMALIZATION_MODE,
        "y_col": y_col,
        "I_pre": I_pre,
        "I_0": fit_params["I_0"],
        "I_inf": fit_params["I_inf"],
        "A": fit_params["A"],
        "k": fit_params["k"],
        "t_half": fit_params["t_half"],
        "absolute_t_half": fit_params["absolute_t_half"],
        "I_half": fit_params["I_half"],
        "mobile_fraction": mobile_fraction,
        "immobile_fraction": immobile_fraction,
        "n_files": group_df[SOURCE_FILE_COL].nunique(),
    }

    return fit_result, fit_curve

