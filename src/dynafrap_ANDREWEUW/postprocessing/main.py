import re

import pandas as pd

from dynafrap_andreweuw.paths import INPUT_FOLDERS
from dynafrap_andreweuw.postprocessing.config import (
    GROUP_COL,
    GROUPS,
    SAVE_FIT_RESULTS,
    SOURCE_FILE_COL,
    fit_results_csv,
)
from dynafrap_andreweuw.postprocessing.plot import plot_groups_combined
from dynafrap_andreweuw.postprocessing.process import (
    correct_target,
    get_y_col,
)


def collect_group_files(input_folders, groups):
    """
    Collect CSV files from all input folders into groups based on
    filename regex matching.
    """
    grouped_files = {group.name: [] for group in groups}

    csv_files = [
        csv_file
        for input_folder in input_folders
        for csv_file in input_folder.glob("*.csv")
    ]

    for csv_file in csv_files:
        for group in groups:
            if re.search(group.pattern, csv_file.name, re.IGNORECASE):
                grouped_files[group.name].append(csv_file)

    return grouped_files


def load_and_process_file(file_path, group_name):
    """
    Load and process one CSV file.
    """
    df = pd.read_csv(file_path)
    df = correct_target(df)

    df[GROUP_COL] = group_name
    df[SOURCE_FILE_COL] = file_path.name

    return df


def load_and_process_all(grouped_files):
    """
    Load, correct, annotate, and concatenate all files.
    """
    all_results = []

    for group_name, files in grouped_files.items():
        print(f"\nProcessing group: {group_name}")

        if len(files) == 0:
            print(f"  No files found for group: {group_name}")
            continue

        for file_path in files:
            print(f"  Processing {file_path}")
            df = load_and_process_file(file_path, group_name)
            all_results.append(df)

    if len(all_results) == 0:
        raise RuntimeError("No CSV files were processed. Check INPUT_FOLDERS and GROUPS.")

    return pd.concat(all_results, ignore_index=True)


# =============================================================================
# Main
# =============================================================================

def main():
    y_col = get_y_col()

    grouped_files = collect_group_files(
        input_folders=INPUT_FOLDERS,
        groups=GROUPS,
    )

    print("Grouped files:")
    for group_name, files in grouped_files.items():
        print(group_name)
        for file_path in files:
            print("  ", file_path)

    all_results = load_and_process_all(grouped_files)

    fit_results_df = plot_groups_combined(
        all_results=all_results,
        y_col=y_col,
    )

    print("\nFit results:")
    print(fit_results_df)

    if SAVE_FIT_RESULTS:
        csv_path = fit_results_csv()
        fit_results_df.to_csv(csv_path, index=False)
        print(f"\nSaved fit results to: {csv_path}")

if __name__ == "__main__":
    main()
