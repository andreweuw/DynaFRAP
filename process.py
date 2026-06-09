import numpy as np
from numpy.typing import NDArray
from skimage.feature import peak_local_max


def detect_local_max(volume):
    return peak_local_max(volume, min_distance=3, exclude_border=True, threshold_rel=0.3)


def pick_channel(volumes, channel):
    volumes = volumes[:, channel]
    return volumes


def normalize_all(volumes):
    normalized_volumes = []
    for volume in volumes:
        # Normalize to 0-1
        v_min, v_max = volume.min(), volume.max()
        volume_norm = (volume - v_min) / (v_max - v_min) if v_max > v_min else np.zeros_like(volume)
        normalized_volumes.append(volume_norm)

    return normalized_volumes


def crop_all(volumes, thresh, pad):
    ymin, xmin, ymax, xmax = None, None, None, None

    all_values = np.concatenate([v.ravel() for v in volumes])  # flatten all volumes
    percentile = 40  # 70th percentile
    thresh = np.percentile(np.unique(all_values.astype(int)), percentile)
    print(np.max(all_values))
    print(thresh)
    for volume in volumes:
        nonzero = np.argwhere(volume > thresh)
        if nonzero.size == 0:
            continue
        _, y0, x0 = nonzero.min(axis=0)
        _, y1, x1 = nonzero.max(axis=0)
        ymin = y0 if ymin is None else min(ymin, y0)
        xmin = x0 if xmin is None else min(xmin, x0)
        ymax = y1 if ymax is None else max(ymax, y1)
        xmax = x1 if xmax is None else max(xmax, x1)
    if ymin is None:
        return volumes
    # Apply padding and clip
    ymin = max(ymin - pad, 0)
    xmin = max(xmin - pad, 0)
    ymax = min(ymax + pad, volumes[0].shape[1] - 1)
    xmax = min(xmax + pad, volumes[0].shape[2] - 1)

    # Crop all frames with the same box
    cropped = [v[:, ymin : ymax + 1, xmin : xmax + 1] for v in volumes]
    return np.array(cropped), np.array([xmin, ymin, 0])


def crop_to_nonzero_bbox(
    volume: np.ndarray, thresh=0.5, pad=10
) -> tuple[NDArray, tuple[float, float, float]]:
    """Crop a 3D volume to the bounding box of voxels > 0."""
    assert volume.ndim == 3, "Expected a 3D volume"

    # Find indices of nonzero voxels
    nonzero = np.argwhere(volume > thresh)
    if nonzero.size == 0:
        return volume, (0, 0, 0)  # nothing nonzero, return as-is

    # Only compute bounding box in Y and X
    _, ymin, xmin = nonzero.min(axis=0)
    _, ymax, xmax = nonzero.max(axis=0)

    # Crop (inclusive ranges, so add +1 on max side)
    ymin = max(ymin - pad, 0)
    ymax = min(ymax + pad, volume.shape[1] - 1)
    xmin = max(xmin - pad, 0)
    xmax = min(xmax + pad, volume.shape[2] - 1)

    # Crop: keep all Z, crop Y and X
    cropped = volume[:, ymin : ymax + 1, xmin : xmax + 1]

    return cropped, (0, ymin, xmin)
