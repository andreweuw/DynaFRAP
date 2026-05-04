import csv
from math import nan
import os
import numpy as np

from . import analysis
from .primitives import add_cylinder


def compute_averages_and_save_to_csv(viewer):
    s = viewer.filenames[0]
    csv_file = f"{s[:s.find('_Time')]}.csv"
    start_t = viewer.current_t
    t0 = viewer.current_t
    new_rows = []
    radius = viewer.radius
    height = viewer.spacing[2] * 3

    for t in range(start_t, viewer.T):
        if t in viewer.cylinder_actors:
            for a in viewer.cylinder_actors[t]:
                viewer.renderer.RemoveActor(a)
            viewer.cylinder_actors[t] = []

        if t not in viewer.target_peaks:
            print(f"No target peak selected for frame {t}.")
            continue

        volume = viewer.volumes[t]
        image = viewer.images[t]
        peak = viewer.target_peaks[t]
        control = viewer.control_peaks.get(t)
        control_center = control if control is not None else ("", "", "")



        import numpy as np
        zc, yc, xc = peak
        print("DEBUG center (peak):", peak)
        print("DEBUG shift (viewer.min):", viewer.min)
        print("DEBUG shapes: image", image.shape, "volume", volume.shape)
        # build same mask _mean_of_cylinder uses (center is z,y,x)
        zz, yy, xx = np.indices(volume.shape)
        half_h = int(height) // 2
        zmin = max(0, zc - half_h)
        zmax = min(volume.shape[0], zc + half_h + 1)
        dist2d = np.sqrt((yy - yc) ** 2 + (xx - xc) ** 2)
        mask = (dist2d <= radius) & (zz >= zmin) & (zz < zmax)
        vals = volume[mask]
        print("DEBUG mask voxels:", mask.sum(), "vals dtype:", vals.dtype, "nan count:", np.isnan(vals).sum(), "sample:", vals.ravel()[:10])

        target_mean, target_mean_likelihood = analysis.cylinder_mean(image, viewer.min, peak, radius, volume, height)
        control_mean, control_mean_likelihood = analysis.cylinder_mean(image, viewer.min, control, radius, volume, height) if control is not None else (nan, nan)

        print(f"Frame {t}: target_mean={target_mean:.3f}, target_mean_likelihood={target_mean_likelihood:.3f}, "
              f"control_mean={control_mean:.3f}, control_mean_likelihood={control_mean_likelihood:.3f}")

        new_rows.append([
            t,
            target_mean,
            control_mean,
            f"[{peak[0]},{peak[1]},{peak[2]}]",
            radius,
            height,
            f"[{control_center[0]},{control_center[1]},{control_center[2]}]" if control is not None else "",
            radius if control is not None else "",
            height if control is not None else "",
            target_mean_likelihood,
            control_mean_likelihood if control is not None else "",
        ])
        print(f"Frame {t}: target={target_mean:.3f}, control_mean={control_mean:.3f}")

        actors = []
        target_actor = add_cylinder(viewer, height, radius, peak, (1, 0, 0), opacity=0.4)
        actors.append(target_actor)
        if control is not None:
            control_actor = add_cylinder(viewer, height, radius, control, (0, 0, 1), opacity=0.4)
            actors.append(control_actor)
        viewer.cylinder_actors[t] = actors

    update_csv(viewer, csv_file, new_rows, t0)
    viewer.update_frame(viewer.current_t)


def update_csv(viewer, csv_file, new_rows, t0):
    header = [
        "time",
        "target_mean",
        "control_mean",
        "target_center[z=depth,y,x]",
        "target_radius",
        "target_height",
        "control_center[z=depth,y,x]",
        "control_radius",
        "control_height",
        "target_mean_likelihood",
        "control_mean_likelihood",
    ]
    row_width = len(header)

    def pad_row(row):
        padded = list(row[:row_width])
        if len(padded) < row_width:
            padded.extend([""] * (row_width - len(padded)))
        return padded

    # Prepare a fixed-size rows buffer with one entry per time frame
    T = viewer.T
    rows = [[""] * row_width for _ in range(T)]

    if os.path.exists(csv_file):
        with open(csv_file, newline="") as f:
            reader = list(csv.reader(f))

        # find header row in existing file
        header_line_idx = None
        for i, row in enumerate(reader):
            if len(row) >= 1 and row[0] == "time":
                header_line_idx = i
                break

        if header_line_idx is None:
            raise RuntimeError(f"CSV {csv_file} exists but has no header 'time' row")

        existing_header = reader[header_line_idx]
        if existing_header != header:
            raise RuntimeError("CSV header mismatch — runtime mismatch detected. Aborting to avoid corrupting existing data.")

        existing_rows = reader[header_line_idx + 1:]
        # populate buffer with existing rows (preserve previous values)
        for i in range(min(len(existing_rows), T)):
            rows[i] = pad_row(existing_rows[i])

    # Overwrite rows for times provided in new_rows (these correspond to times t..T-1)
    for r in new_rows:
        if not r:
            continue
        time_val = r[0]
        try:
            ti = int(time_val)
        except Exception:
            continue
        if 0 <= ti < T:
            rows[ti] = pad_row(r)

    # Format numeric cells (floats) to 5 decimal places; treat NaN as empty
    def _format_cell(v):
        try:
            if isinstance(v, str):
                return v
            # handle numpy scalar types
            if isinstance(v, (np.floating, float)):
                if np.isnan(v):
                    return ""
                return f"{float(v):.5f}"
            if isinstance(v, (np.integer, int)):
                return str(int(v))
        except Exception:
            pass
        return v

    rows_formatted = [[_format_cell(c) for c in row] for row in rows]

    # Write file: header, then exactly T rows (indexed by time)
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows_formatted)

    print(f"CSV updated from frame {t0} onwards → {csv_file}")