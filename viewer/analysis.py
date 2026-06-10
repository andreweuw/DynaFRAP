import numpy as np
from skimage.feature import peak_local_max


def compute_target_peak(volume, mask=None):
    masked_volume = np.where(mask, volume, -np.inf) if mask is not None else volume
    flat_idx = np.argmax(masked_volume)
    return np.unravel_index(flat_idx, volume.shape)


def compute_target_peak2(
    volume,
    mask=None,
    prev_peak=None,
    min_distance=2,
    num_candidates=5,
    intensity_weight=1.0,
    distance_penalty=0.2,
    jump_penalty=0.0,
):
    search_vol = np.where(mask, volume, -np.inf) if mask is not None else volume
    candidates = peak_local_max(
        search_vol,
        min_distance=min_distance,
        num_peaks=num_candidates,
    )
    if len(candidates) == 0:
        return (
            prev_peak
            if prev_peak is not None
            else np.unravel_index(np.argmax(search_vol), volume.shape)
        )
    scores = []
    for z, y, x in candidates:
        intensity = volume[z, y, x]
        if prev_peak is not None:
            dz, dy, dx = np.array([z, y, x]) - np.array(prev_peak)
            dist = np.sqrt(dz**2 + dy**2 + dx**2)
            prev_int = volume[tuple(prev_peak)]
            jump = abs(intensity - prev_int)
        else:
            dist = 0.0
            jump = 0.0

        score = intensity_weight * intensity - distance_penalty * dist - jump_penalty * jump
        scores.append(score)

    best_idx = int(np.argmax(scores))
    return tuple(candidates[best_idx])


def mask_from_bounds(bounds, shape, spacing):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    zi_min = int(np.clip(zmin / spacing[2], 0, shape[0] - 1))
    zi_max = int(np.clip(zmax / spacing[2], 0, shape[0] - 1))
    yi_min = int(np.clip(ymin / spacing[1], 0, shape[1] - 1))
    yi_max = int(np.clip(ymax / spacing[1], 0, shape[1] - 1))
    xi_min = int(np.clip(xmin / spacing[0], 0, shape[2] - 1))
    xi_max = int(np.clip(xmax / spacing[0], 0, shape[2] - 1))

    mask = np.zeros(shape, dtype=bool)
    mask[zi_min : zi_max + 1, yi_min : yi_max + 1, xi_min : xi_max + 1] = True
    return mask


def propagate_peak_bounds(bounds, peak_next, origin, spacing):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    px = origin[0] + (peak_next[2] + 0.5) * spacing[0]
    py = origin[1] + (peak_next[1] + 0.5) * spacing[1]

    dx, dy = px - cx, py - cy
    return [xmin + dx, xmax + dx, ymin + dy, ymax + dy, zmin, zmax]


def logits_to_probability_100(logits):
    return 1.0 / (1.0 + np.exp(-logits)) * 100


def _mean_of_cylinder(ndarray, center, radius, height):
    # center is (z, y, x)
    zz, yy, xx = np.indices(ndarray.shape)
    zc = int(center[0])
    yc = center[1]
    xc = center[2]
    dist2d = np.sqrt((yy - yc) ** 2 + (xx - xc) ** 2)
    half_h = height // 2
    zmin = int(max(0, zc - half_h))
    zmax = int(min(ndarray.shape[0], zc + half_h + 1))
    mask = (dist2d <= radius) & (zz >= zmin) & (zz < zmax)
    vals = ndarray[mask]
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()) if vals.size > 0 else np.nan


def cylinder_mean(image, shift, center, radius, volume, height):
    # Mean likelihood in NNUnet logits likelihood using cylindrical ROI
    likelihood_mean = _mean_of_cylinder(volume, center, radius, height)
    # Convert logits to 0-100 probability scale
    likelihood_mean = logits_to_probability_100(likelihood_mean)

    # Mean intensity in image using cylindrical ROI
    adjusted_center = center + np.array([shift[2], shift[1], shift[0]])
    intensity_mean = _mean_of_cylinder(image, adjusted_center, radius, height)
    return intensity_mean, likelihood_mean


def compute_volume_of_target_time_zero(volume, peak, radius=4, height=3):
    segmentation = volume > 0.5
    zc, yc, xc = peak

    half_h = height // 2
    zmin = max(0, zc - half_h)
    zmax = min(volume.shape[0], zc + half_h + 1)

    zz, yy, xx = np.indices(volume.shape)
    dist2d = np.sqrt((yy - yc) ** 2 + (xx - xc) ** 2)
    cyl_mask = (dist2d <= radius) & (zz >= zmin) & (zz <= zmax)

    seg_cyl = segmentation & cyl_mask
    return int(np.sum(seg_cyl))


def estimate_radius_at_frame_zero(volume, image, bounds, shape, spacing, shift):
    """
    Find peak in selection box and estimate radius where probability of cell is at least 80%.

    Args:
        volume: segmentation/likelihood volume at frame 0
        image: image volume at frame 0
        bounds: box bounds [xmin, xmax, ymin, ymax, zmin, zmax]
        shape: volume shape
        spacing: voxel spacing
        shift: image shift offset

    Returns:
        tuple: (peak, estimated_radius, likelihood) or (peak, None, None) if unsuccessful
    """
    mask = mask_from_bounds(bounds, shape, spacing)
    peak_t = compute_target_peak(volume, mask=mask)

    if peak_t is None:
        print("Could not find peak in selection box")
        return peak_t, None, None

    print(f"Found peak at {peak_t}")

    # Try radii in steps, looking for largest radius with likelihood >= 80%
    radius_min = 1.0
    radius_max = 20.0
    height = 3

    best_radius = None
    best_likelihood = 0.0

    for test_radius in np.linspace(radius_min, radius_max, 40):
        intensity_mean, likelihood_mean = cylinder_mean(
            image, shift, peak_t, test_radius, volume, height
        )

        if np.isnan(likelihood_mean):
            continue

        print(f"  Radius {test_radius:.2f}: likelihood = {likelihood_mean:.1f}%")

        if likelihood_mean >= 80.0:
            best_radius = test_radius
            best_likelihood = likelihood_mean

    if best_radius is not None:
        print(f"✓ Estimated radius: {best_radius:.2f} pixels (likelihood: {best_likelihood:.1f}%)")
    else:
        print("✗ Could not find radius with 80% likelihood")

    return peak_t, best_radius, best_likelihood
