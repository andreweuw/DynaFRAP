# =============================================================================
# Plotting functions
# =============================================================================

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dynafrap_andreweuw.post_analysis.config import (
    CI_MULTIPLIER,
    FIGSIZE,
    FIT_LINE_WIDTH,
    GROUP_COL,
    GROUP_COLORS,
    GROUP_NAMES,
    INDIVIDUAL_LINE_ALPHA,
    INDIVIDUAL_LINE_WIDTH,
    MEAN_LINE_WIDTH,
    SHOW_CI,
    SHOW_HALFTIME_MARKER,
    SHOW_INDIVIDUAL_TRACES,
    SOURCE_FILE_COL,
    TIME_COL,
    X_LABEL,
    Y_LABEL,
    TITLE,
)
from dynafrap_andreweuw.post_analysis.process import (
    fit_group_mean,
    get_y_col,
)


def summarize_group(group_df, y_col):
    """
    Compute mean, std, count, SEM, and CI per time point for one group.
    """
    summary = (
        group_df.groupby(TIME_COL)[y_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(TIME_COL)
    )

    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci"] = CI_MULTIPLIER * summary["sem"]

    return summary

def plot_groups_combined(all_results, y_col=None):
    """
    Plot all groups in one chart.

    Shows:
        - faint individual sample traces
        - group mean
        - 95% CI
        - exponential fit
        - t_1/2 marker
    """
    if y_col is None:
        y_col = get_y_col()

    plt.figure(figsize=FIGSIZE)

    fit_results = []

    for group_name in GROUP_NAMES:
        group_df = all_results[all_results[GROUP_COL] == group_name].copy()

        if group_df.empty:
            print(f"No data for group: {group_name}")
            continue

        color = GROUP_COLORS.get(group_name, None)

        if SHOW_INDIVIDUAL_TRACES:
            plot_individual_traces(group_df, y_col, color)

        summary = summarize_group(group_df, y_col)

        fit_result, fit_curve = fit_group_mean(
            summary=summary,
            group_df=group_df,
            group_name=group_name,
            y_col=y_col,
        )

        plot_group_mean_and_ci(
            summary=summary,
            group_name=group_name,
            color=color,
        )

        if fit_result is not None and fit_curve is not None:
            plot_fit_curve(
                fit_curve=fit_curve,
                fit_result=fit_result,
                group_name=group_name,
                color=color,
            )

            fit_results.append(fit_result)

    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.title(TITLE)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(fit_results)


def plot_individual_traces(group_df, y_col, color):
    """
    Plot individual sample traces for one group.
    """
    for _, file_df in group_df.groupby(SOURCE_FILE_COL):
        file_df = file_df.sort_values(TIME_COL)

        plt.plot(
            file_df[TIME_COL],
            file_df[y_col],
            linewidth=INDIVIDUAL_LINE_WIDTH,
            alpha=INDIVIDUAL_LINE_ALPHA,
            color=color,
            label="_nolegend_",
        )


def plot_group_mean_and_ci(summary, group_name, color):
    """
    Plot group mean and confidence interval.
    """
    x = summary[TIME_COL].to_numpy(dtype=float)
    mean = summary["mean"].to_numpy(dtype=float)
    ci = summary["ci"].to_numpy(dtype=float)

    if SHOW_CI:
        plt.fill_between(
            x,
            mean - ci,
            mean + ci,
            alpha=0.15,
            color=color,
            label=f"{group_name} 95% CI",
        )

    plt.plot(
        x,
        mean,
        marker="o",
        linewidth=MEAN_LINE_WIDTH,
        color=color,
        label=f"{group_name} mean",
    )


def plot_fit_curve(fit_curve, fit_result, group_name, color):
    """
    Plot fitted exponential recovery curve and t_1/2 markers.
    """
    plt.plot(
        fit_curve[TIME_COL],
        fit_curve["fit"],
        linestyle="--",
        linewidth=FIT_LINE_WIDTH,
        color=color,
        label=f"{group_name} fit, t1/2={fit_result['t_half']:.2f}",
    )

    if SHOW_HALFTIME_MARKER:
        plt.axvline(
            fit_result["absolute_t_half"],
            linestyle=":",
            linewidth=1.5,
            color=color,
            label=f"{group_name} t1/2 at {fit_result['absolute_t_half']:.2f}",
        )

        plt.axhline(
            fit_result["I_half"],
            linestyle=":",
            linewidth=1.0,
            color=color,
            alpha=0.5,
            label="_nolegend_",
        )
