import os
import re

def fix_filenames(folder, dry_run=True):
    """
    Fix filenames according to rules:
    - If "Pre" is in the filename:
        * Increase series number by +1
        * Reset time number to 0
    - Otherwise keep unchanged.

    Args:
        folder (str): Path to folder with files
        dry_run (bool): If True, only print changes. If False, rename files.
    """
    pattern = re.compile(
        r"(.*Series)(\d+)(_Channel_\d+_Time_)(\d+)(\.tif)$"
    )

    for fname in os.listdir(folder):
        if not fname.endswith(".tif"):
            continue

        match = pattern.match(fname)
        if not match:
            print(f"Skipping: {fname}")
            continue

        prefix, series, middle, time, ext = match.groups()
        series, time = int(series), int(time)

        if "Pre" in fname and time == 1:
            new_series = series - 1
            new_time = 0
        else:
            continue

        new_fname = f"{prefix}{new_series}{middle}{new_time}{ext}"

        old_path = os.path.join(folder, fname)
        new_path = os.path.join(folder, new_fname)

        if dry_run:
            print(f"{fname}  -->  {new_fname}")
        else:
            os.rename(old_path, new_path)


# Example usage:
bases = ['res/', 'flat/']
for base in bases:
    for folder in os.listdir(base):
        fix_filenames(base+folder, dry_run=False)  # set dry_run=False to actually rename