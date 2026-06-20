import numpy as np

from dynafrap_andreweuw.post_analysis.config import (
    CONTROL_BLEACH_FACTOR_COL,
    CONTROL_COL,
    CORRECTED_TARGET_RECOVERY_PERCENT_COL,
    I_PRE_COL,
    MIN_PLOT_TIME,
    PREBLEACH_TIME,
    SOURCE_FILE_COL,
    TARGET_COL,
    TIME_COL,
)
from dynafrap_andreweuw.post_analysis.frap_math import (
    fit_recovery_curve,
    mobile_fraction_from_fit,
)

# =============================================================================
# Column / label helpers
# =============================================================================


def get_y_col():
    return CORRECTED_TARGET_RECOVERY_PERCENT_COL


def get_I_pre_for_fit():
    """
    In the final normalized recovery scale, pre-bleach is always 100%.
    """
    return 100.0


# =============================================================================
# Data processing functions
# =============================================================================


def correct_target(df):
    """
    Apply the final FRAP postprocessing approach to one sample dataframe.

    Steps:

    1. Sort frames by time/frame index.

    2. Find the pre-bleach frame.

    3. Find the immediate post-bleach frame.

    4. Estimate the control bleaching factor:

           control_bleach_factor(t) = control(t) / control(prebleach)

    5. Correct the target intensity for acquisition/photobleaching effects:

           target_corrected(t) = target(t) / control_bleach_factor(t)

    6. Store corrected pre-bleach intensity:

           I_pre = target_corrected(prebleach)

    7. Store corrected immediate post-bleach intensity:

           I_post = target_corrected(first frame after prebleach)

    8. Normalize recovery:

           target_corrected_recovery_percent(t)
               = (target_corrected(t) - I_post) / (I_pre - I_post) * 100

       Therefore:
           pre-bleach = 100%
           immediate post-bleach = 0%

    9. Return only frames from MIN_PLOT_TIME onward.
    """
    df = df.copy()
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    prebleach_rows = df[df[TIME_COL] == PREBLEACH_TIME]

    if prebleach_rows.empty:
        raise ValueError(
            f"No pre-bleach frame found at {TIME_COL} == {PREBLEACH_TIME}"
        )

    prebleach_idx = prebleach_rows.index[0]

    postbleach_rows = df[df[TIME_COL] > PREBLEACH_TIME]

    if postbleach_rows.empty:
        raise ValueError(
            f"No post-bleach frame found after {TIME_COL} == {PREBLEACH_TIME}"
        )

    postbleach_idx = postbleach_rows.index[0]

    target = df[TARGET_COL]
    control = df[CONTROL_COL]

    control_prebleach = control.iloc[prebleach_idx]

    if control_prebleach == 0 or not np.isfinite(control_prebleach):
        raise ValueError(
            "Pre-bleach control intensity is zero or invalid; "
            "cannot perform photobleaching correction."
        )

    control_bleach_factor = control / control_prebleach
    target_corrected = target / control_bleach_factor

    I_pre = target_corrected.iloc[prebleach_idx]
    I_post = target_corrected.iloc[postbleach_idx]

    recovery_denominator = I_pre - I_post

    if recovery_denominator <= 0 or not np.isfinite(recovery_denominator):
        target_corrected_recovery_percent = np.nan
    else:
        target_corrected_recovery_percent = (
            (target_corrected - I_post) / recovery_denominator
        ) * 100.0

    df[CONTROL_BLEACH_FACTOR_COL] = control_bleach_factor
    df[I_PRE_COL] = I_pre
    df[CORRECTED_TARGET_RECOVERY_PERCENT_COL] = target_corrected_recovery_percent

    return df[df[TIME_COL] >= MIN_PLOT_TIME].copy()


def fit_group_mean(summary, group_df, group_name, y_col):
    """
    Fit one-phase exponential recovery to the group mean curve.

    Since the only remaining postprocessing approach normalizes recovery such that
    pre-bleach = 100% and immediate post-bleach = 0%, I_pre for the mobile fraction
    calculation is fixed at 100.
    """
    fit_params, fit_curve = fit_recovery_curve(
        time_values=summary[TIME_COL],
        intensity_values=summary["mean"],
    )

    if fit_params is None:
        print(f"Fit failed or not enough points for group: {group_name}")
        return None, None

    I_pre = get_I_pre_for_fit()

    mobile_fraction, immobile_fraction = mobile_fraction_from_fit(
        I_pre=I_pre,
        I_0=fit_params["I_0"],
        A=fit_params["A"],
    )

    fit_result = {
        "group": group_name,
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
