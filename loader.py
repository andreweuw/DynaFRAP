import os
import re
from tifffile import imread
import numpy as np


def fix_filenames(folder, dry_run=True):
    """
    Fix filenames so Pre snapshots align with their Pb1 series.

    Rules:
    - If a Pre file has a matching Pb1 file (same prefix/channel),
      copy the Pb1 series index and set Pre time to 0.
    - If no Pb1 match is found, fall back to legacy behavior:
      Pre + Time_1 -> Series+1 and Time_0.
    - Other files are unchanged.

    Args:
        folder (str): Path to folder with files
        dry_run (bool): If True, only print changes. If False, rename files.
    """
    pattern = re.compile(
        r"^(?P<prefix>.*?)(?P<label>Pre|Pb1)\s+Series(?P<series>\d+)(?P<middle>_Channel_(?P<channel>\d+)_Time_)(?P<time>\d+)(?P<ext>\.tif)$"
    )

    parsed = []
    for fname in os.listdir(folder):
        if not fname.endswith(".tif") or fname[0] == '.':
            continue

        match = pattern.match(fname)
        if not match:
            print(f"Skipping: {fname}")
            continue
        parsed.append((fname, match))

    pb1_by_key = {}
    for fname, match in parsed:
        gd = match.groupdict()
        if gd["label"] != "Pb1":
            continue
        key = (gd["prefix"].rstrip(" _-"), gd["channel"])
        pb1_by_key.setdefault(key, []).append((fname, int(gd["series"])))

    for fname, match in parsed:
        gd = match.groupdict()
        if gd["label"] != "Pre":
            continue

        series = int(gd["series"])
        time = int(gd["time"])
        key = (gd["prefix"].rstrip(" _-"), gd["channel"])

        new_series = None
        new_time = None

        pb1_candidates = pb1_by_key.get(key, [])
        if len(pb1_candidates) == 1:
            new_series = pb1_candidates[0][1]
            new_time = 0
        elif len(pb1_candidates) > 1:
            # If duplicates exist, choose the closest series index.
            closest = min(pb1_candidates, key=lambda x: abs(x[1] - series))
            new_series = closest[1]
            new_time = 0
        elif time == 1:
            # Legacy fallback for old one-step offset naming.
            print('Warning: No Pb1 match for {}, applying legacy fix.'.format(fname))
            new_series = series + 1
            new_time = 0
        else:
            continue

        if series == new_series and time == new_time:
            continue

        new_fname = f"{gd['prefix']}Pre Series{new_series}{gd['middle']}{new_time}{gd['ext']}"
        old_path = os.path.join(folder, fname)
        new_path = os.path.join(folder, new_fname)

        if os.path.exists(new_path) and old_path != new_path:
            print(f"Skipping (target exists): {fname}  -->  {new_fname}")
            continue

        if dry_run:
            print(f"{fname}  -->  {new_fname}")
        else:
            os.rename(old_path, new_path)

def normalize_prefix(prefix: str) -> str:
    """
    Normalize prefix by stripping ignorable tags like 'Pre' or 'Pb'
    and trailing underscores/hyphens.
    """
    bad_tokens = {"Pre", "Pb1"}
    # Split prefix on underscores/dashes
    tokens = re.split(r"[\s_\-]+", prefix)
    tokens = [t for t in tokens if t and t not in bad_tokens]
    return "_".join(tokens) if tokens else ""

def load_dual_3d_tiff_sequence(folder_logits, folder_images, pb1_max_frame=None):
    """
    Load paired 3D TIFF volumes: logits + raw images.
    Filenames must match between the two folders.
    Args:
        folder_logits (str): Path to logits TIFFs.
        folder_images (str): Path to raw image TIFFs.
        pb1_max_frame (int | None): If set, only load Pb1 frames with
            Time_ <= pb1_max_frame. Pre frames are kept.

    Returns:
        all_logits, all_images, all_filenames
    where each entry is (T, C, Z, Y, X).
    """
    fix_filenames(folder_logits, dry_run=False)
    fix_filenames(folder_images, dry_run=False)
    import os, re
    import numpy as np
    from tifffile import imread

    pattern = re.compile(r"(.+?)Series(\d+)_Channel_\d+_Time_(\d+)\.tif$")

    file_entries = []
    for fname in os.listdir(folder_logits):
        if not fname.endswith(".tif"):
            continue
        match = pattern.search(fname)
        if match:
            raw_prefix = match.group(1)
            prefix = normalize_prefix(raw_prefix)
            series_idx = int(match.group(2))
            time_idx = int(match.group(3))
            if pb1_max_frame is not None and " Pb1 " in fname and time_idx > pb1_max_frame:
                continue
            file_entries.append((prefix, series_idx, time_idx, fname))
        else:
            print(f"Skipping: {fname}")

    # sort by prefix, series, time
    file_entries.sort(key=lambda x: (x[0], x[1], x[2]))

    # group by (prefix, series)
    series_map = {}
    for prefix, series, time, fname in file_entries:
        series_map.setdefault((prefix, series), []).append(fname)

    all_logits = []
    all_images = []
    all_filenames = []

    for (prefix, series) in sorted(series_map.keys()):
        fnames = series_map[(prefix, series)]

        logits_vols, image_vols = [], []
        for f in fnames:
            # load logits
            v1 = imread(os.path.join(folder_logits, f))
            if v1.shape[1] > 13:
                v1 = v1[:, :13, :, :]
            elif v1.shape[1] < 13:
                raise ValueError(f"Logits {f} has too few slices: {v1.shape}")
            logits_vols.append(v1)

            # load matching raw image
            f_img = os.path.join(folder_images, f)
            if not os.path.exists(f_img):
                f_img = os.path.join(folder_images, f.replace('Series', 'Series0'))
                if not os.path.exists(f_img):
                    raise FileNotFoundError(f"Missing raw image: {f_img}")
            v2 = imread(f_img)
            if v2.shape[0] > 13:
                v2 = v2[:13, :, :]
            elif v2.shape[0] < 13:
                raise ValueError(f"Image {f} has too few slices: {v2.shape}")
            image_vols.append(v2)

        logits_vols = np.stack(logits_vols, axis=0)  # (T, C, Z, Y, X)
        image_vols = np.stack(image_vols, axis=0)    # (T, Z, Y, X)

        all_logits.append(logits_vols)
        all_images.append(image_vols)
        all_filenames.append(fnames)

    return all_logits, all_images, all_filenames

def load_3d_tiff_sequence(folder_path, pb1_max_frame=None):
    """
    Load 3D TIFF series grouped by (normalized prefix, series).

    Args:
        folder_path (str): Path to TIFF folder.
        pb1_max_frame (int | None): If set, only load Pb1 frames with
            Time_ <= pb1_max_frame. Pre frames are kept.
    """
    fix_filenames(folder_path, dry_run=False)
    pattern = re.compile(r"(.+?)Series(\d+)_Channel_\d+_Time_(\d+)\.tif$")

    file_entries = []
    for fname in os.listdir(folder_path):
        if not fname.endswith(".tif"):
            continue
        match = pattern.search(fname)
        if match:
            raw_prefix = match.group(1) # text before 'Series'
            prefix = normalize_prefix(raw_prefix)
            series_idx = int(match.group(2))
            time_idx = int(match.group(3))
            if pb1_max_frame is not None and " Pb1 " in fname and time_idx > pb1_max_frame:
                continue
            file_entries.append((prefix, series_idx, time_idx, fname))
        else:
            print(f"Skipping: {fname}")

    # sort by prefix, series, then time
    file_entries.sort(key=lambda x: (x[0], x[1], x[2]))

    # group by (prefix, series)
    series_map = {}
    for prefix, series, time, fname in file_entries:
        series_map.setdefault((prefix, series), []).append(fname)

    all_volumes = []
    all_filenames = []

    for (prefix, series) in sorted(series_map.keys()):
        fnames = series_map[(prefix, series)]
        volumes = []
        for f in fnames:
            v = imread(os.path.join(folder_path, f))  # shape (C, Z, Y, X)
            if v.shape[1] > 13:  # crop extra Z slices
                v = v[:, :13, :, :]
            elif v.shape[1] < 13:
                raise ValueError(f"Volume {f} has too few slices: {v.shape}")
            volumes.append(v)
        volumes = np.stack(volumes, axis=0)  # (T, C, Z, Y, X)
        all_volumes.append(volumes)
        all_filenames.append(fnames)

    return all_volumes, all_filenames


    files = []

    for fname in os.listdir(folder_path):
        match = pattern.search(fname)
        print(match, fname)
        if match:
            time_idx = int(match.group(1))
            files.append((time_idx, fname))

    files.sort(key=lambda x: (x[0], x[1]))
    filenames = [fname for _, fname in files]
    volumes = [imread(os.path.join(folder_path, fname)) for fname in filenames]

    # now volumes is a list of arrays of shape (C, Z, Y, X)
    volumes = np.stack(volumes)  # (T, C, Z, Y, X)
    return volumes, filenames

def compute_global_bounding_box(volumes, channel=0):
    T, C, Z, Y, X = volumes.shape
    min_z, min_y, min_x = Z, Y, X
    max_z, max_y, max_x = 0, 0, 0

    for t in range(T):
        mask = volumes[t, channel] > 0  # pick one channel
        coords = np.argwhere(mask)
        if coords.size == 0:
            continue
        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0)
        min_z, min_y, min_x = min(min_z, z0), min(min_y, y0), min(min_x, x0)
        max_z, max_y, max_x = max(max_z, z1), max(max_y, y1), max(max_x, x1)
    bbox = slice(min_z, max_z + 1), slice(min_y, max_y + 1), slice(min_x, max_x + 1)
    return bbox, np.array([min_z,min_y,min_x])

def crop_volumes_to_bounding_box(volumes, bbox):
    z_slice, y_slice, x_slice = bbox
    return volumes[:, :, z_slice, y_slice, x_slice]
