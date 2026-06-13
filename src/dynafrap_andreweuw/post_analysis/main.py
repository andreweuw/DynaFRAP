import pandas as pd

from dynafrap_andreweuw.post_analysis.config import (
    FIT_RESULTS_CSV,
    GROUP_COL,
    GROUP_NAMES,
    INPUT_FOLDER,
    SAVE_FIT_RESULTS,
    SOURCE_FILE_COL,
)
from dynafrap_andreweuw.post_analysis.plot import plot_groups_combined
from dynafrap_andreweuw.post_analysis.process import (
    correct_target,
    get_y_col_from_normalization_mode,
)


def collect_group_files(input_folder, group_names):
    """
    Collect CSV files into groups based on filename matching.
    """
    grouped_files = {group: [] for group in group_names}

    csv_files = list(input_folder.glob("*.csv"))

    for csv_file in csv_files:
        for group in group_names:
            if group.lower() in csv_file.name.lower():
                grouped_files[group].append(csv_file)

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
        raise RuntimeError("No CSV files were processed. Check INPUT_FOLDER and GROUP_NAMES.")

    return pd.concat(all_results, ignore_index=True)


# =============================================================================
# Main
# =============================================================================

def main():
    y_col = get_y_col_from_normalization_mode()

    grouped_files = collect_group_files(
        input_folder=INPUT_FOLDER,
        group_names=GROUP_NAMES,
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
        fit_results_df.to_csv(FIT_RESULTS_CSV, index=False)
        print(f"\nSaved fit results to: {FIT_RESULTS_CSV}")

if __name__ == "__main__":
    main()
