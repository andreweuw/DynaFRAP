# =============================================================================
# Plotting functions
# =============================================================================

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from dynafrap_andreweuw.postprocessing.config import (
    CI_MULTIPLIER,
    FIGSIZE,
    FIT_LINE_WIDTH,
    GROUP_COL,
    GROUPS,
    INDIVIDUAL_LINE_ALPHA,
    INDIVIDUAL_LINE_WIDTH,
    MEAN_LINE_WIDTH,
    PLOT_OUTLIERS,
    SHOW_CI,
    SHOW_HALFTIME_MARKER,
    SHOW_INDIVIDUAL_TRACES,
    SOURCE_FILE_COL,
    TIME_COL,
    TITLE,
    X_LABEL,
    Y_LABEL,
)
from dynafrap_andreweuw.postprocessing.process import (
    fit_group_mean,
    get_y_col,
    resolve_outliers,
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
    plotted_groups = []

    for group in GROUPS:
        group_name = group.name
        group_df = all_results[all_results[GROUP_COL] == group_name].copy()

        if group_df.empty:
            print(f"No data for group: {group_name}")
            continue

        color = group.color
        plotted_groups.append((group_name, color))

        stats_df, excluded_df, n_outliers = resolve_outliers(group_df, y_col)

        if n_outliers > 0:
            print(f"{group_name}: {n_outliers} outlier file(s) detected")

        if SHOW_INDIVIDUAL_TRACES:
            plot_individual_traces(stats_df, y_col, color)

        if PLOT_OUTLIERS and not excluded_df.empty:
            plot_excluded_traces(excluded_df, y_col, color)

        summary = summarize_group(stats_df, y_col)

        fit_result, fit_curve = fit_group_mean(
            summary=summary,
            group_df=stats_df,
            group_name=group_name,
            y_col=y_col,
            n_outliers=n_outliers,
        )

        plot_group_mean_and_ci(
            summary=summary,
            color=color,
        )

        if fit_result is not None and fit_curve is not None:
            plot_fit_curve(
                fit_curve=fit_curve,
                fit_result=fit_result,
                color=color,
            )

            fit_results.append(fit_result)

    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.title(TITLE)
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))

    style_handles = [
        Line2D([0], [0], color="black", marker="o", linewidth=MEAN_LINE_WIDTH, label="Mean"),
    ]
    if SHOW_CI:
        style_handles.append(Patch(facecolor="black", alpha=0.15, label="95% CI (area)"))
    style_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=FIT_LINE_WIDTH,
            label=r"Fit: $I(t) = I_{\infty}\left(1 - e^{-kt}\right),\ I(0) = 0$",
        )
    )
    if SHOW_HALFTIME_MARKER:
        style_handles.append(
            Line2D([0], [0], color="black", linestyle=":", linewidth=1.5, label=r"$t_{1/2}$")
        )

    group_handles = [
        Line2D([0], [0], color=color, linewidth=MEAN_LINE_WIDTH, label=group_name)
        for group_name, color in plotted_groups
    ]

    spacer = Line2D([0], [0], color="none", label="")
    plt.legend(
        handles=[*group_handles, spacer, *style_handles],
        loc="upper left",
        ncol=1,
    )

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


def plot_excluded_traces(excluded_df, y_col, color):
    """
    Plot the full raw trace of each file excluded from the group mean/CI/fit
    due to outlier frames (see CORRECT_OUTLIER / PLOT_OUTLIERS in config),
    so the excluded data stays visible instead of silently disappearing.
    """
    for _, file_df in excluded_df.groupby(SOURCE_FILE_COL):
        file_df = file_df.sort_values(TIME_COL)

        plt.plot(
            file_df[TIME_COL],
            file_df[y_col],
            linestyle=":",
            linewidth=INDIVIDUAL_LINE_WIDTH,
            alpha=0.7,
            color=color,
            label="_nolegend_",
        )


def plot_group_mean_and_ci(summary, color):
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
            alpha=0.25,
            color=color,
            label="_nolegend_",
        )

    plt.plot(
        x,
        mean,
        marker="o",
        linewidth=MEAN_LINE_WIDTH,
        color=color,
        label="_nolegend_",
    )


def plot_fit_curve(fit_curve, fit_result, color):
    """
    Plot fitted exponential recovery curve and t_1/2 markers.
    """
    plt.plot(
        fit_curve[TIME_COL],
        fit_curve["fit"],
        linestyle="--",
        linewidth=FIT_LINE_WIDTH,
        color=color,
        label="_nolegend_",
    )

    if SHOW_HALFTIME_MARKER:
        plt.axvline(
            fit_result["absolute_t_half"],
            linestyle=":",
            linewidth=1.5,
            color=color,
            label="_nolegend_",
        )

        plt.axhline(
            fit_result["I_half"],
            linestyle=":",
            linewidth=1.0,
            color=color,
            alpha=0.5,
            label="_nolegend_",
        )
