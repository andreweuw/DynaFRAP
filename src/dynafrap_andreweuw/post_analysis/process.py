import numpy as np

from dynafrap_andreweuw.post_analysis.config import (
    CONTROL_BLEACH_FACTOR_COL,
    CONTROL_BLEACH_FACTOR_OUTLIER_THRESHOLD,
    CONTROL_COL,
    CORRECT_OUTLIER,
    CORRECTED_TARGET_RECOVERY_PERCENT_COL,
    I_PRE_COL,
    IS_OUTLIER_COL,
    MAX_PLOT_TIME,
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
    df[IS_OUTLIER_COL] = control_bleach_factor < CONTROL_BLEACH_FACTOR_OUTLIER_THRESHOLD
    df[I_PRE_COL] = I_pre
    df[CORRECTED_TARGET_RECOVERY_PERCENT_COL] = target_corrected_recovery_percent

    df = df[df[TIME_COL] >= MIN_PLOT_TIME]

    if MAX_PLOT_TIME is not None:
        df = df[df[TIME_COL] <= MAX_PLOT_TIME]

    return df.copy()


def resolve_outliers(group_df, y_col):
    """
    Apply outlier handling to one group's combined dataframe.

    A file is considered an outlier file if any of its frames were flagged
    in IS_OUTLIER_COL (control channel collapsed, corrected intensity
    unreliable).

    If CORRECT_OUTLIER is True, outlier frames are inpainted with the
    group's per-timepoint mean computed from non-outlier frames, and the
    file is kept in the returned stats dataframe.

    If CORRECT_OUTLIER is False, outlier files are removed entirely from
    the stats dataframe and returned separately so callers can still plot
    their raw trace.

    Returns:
        stats_df: dataframe to use for the group mean/CI/fit.
        excluded_df: outlier-file rows to plot separately (empty unless
            CORRECT_OUTLIER is False).
        n_outliers: number of distinct files that had at least one outlier
            frame, regardless of CORRECT_OUTLIER.
    """
    outlier_files = group_df.loc[group_df[IS_OUTLIER_COL], SOURCE_FILE_COL].unique()
    n_outliers = len(outlier_files)

    if CORRECT_OUTLIER:
        clean_df = group_df[~group_df[IS_OUTLIER_COL]]

        if clean_df.empty:
            return group_df, group_df.iloc[0:0], n_outliers

        clean_means = clean_df.groupby(TIME_COL)[y_col].mean()

        stats_df = group_df.copy()
        outlier_mask = stats_df[IS_OUTLIER_COL]
        stats_df.loc[outlier_mask, y_col] = stats_df.loc[outlier_mask, TIME_COL].map(clean_means)

        return stats_df, group_df.iloc[0:0], n_outliers

    is_outlier_file = group_df[SOURCE_FILE_COL].isin(outlier_files)

    return group_df[~is_outlier_file], group_df[is_outlier_file], n_outliers


def fit_group_mean(summary, group_df, group_name, y_col, n_outliers=0):
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
        "n_outliers": n_outliers,
    }

    return fit_result, fit_curve
