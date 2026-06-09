from pathlib import Path

from loader import (
    compute_global_bounding_box,
    load_dual_3d_tiff_sequence,
)
from process import crop_all, pick_channel
from viewer.voxel_viewer import VoxelViewer


def process_all_from_folders(folder_path, gt_path, pb1_max_frame=None):
    # volumes, filenames = load_3d_tiff_sequence(folder_path)
    all_volumes, all_images, all_filenames = load_dual_3d_tiff_sequence(
        folder_path,
        gt_path,
        pb1_max_frame=pb1_max_frame,
    )
    for volumes, filenames, images in zip(all_volumes, all_filenames, all_images, strict=False):
        print(volumes.shape)

        bbox, _ = compute_global_bounding_box(volumes)
        # volumes = crop_volumes_to_bounding_box(volumes, bbox)

        volumes = pick_channel(volumes, 1)
        volumes, min = crop_all(volumes, 0.99, 20)
        viewer = VoxelViewer(
            volumes, filenames, images, min, spacing=(1, 1, 3)
        )  # adjust voxel spacing if needed
        viewer.start()


if __name__ == "__main__":
    # Do we know how many one palisade cell can have?
    # can some appear and some controls vanish?

    # inspect

    # Generate results for the different light conditions, make cylinders less high
    # freddy reinvestigates the plots
    # richer output

    # article doesnt have to be before the bigger paper, can be inconporated into the image analysis platform

    res_dir = Path("res")
    flat_dir = Path("flat")
    mapping = {}
    exclude_containing = [
        "Guard-1_5",
        "High-Red",
        "Low-Red-Guard-1-1",
        "Low-Red-Guard-1-2",
        "Low-Red-Pavement-003",
    ]
    for res_item in res_dir.iterdir():
        name = res_item.name
        skip = False
        # for ex in exclude_containing:
        #     if ex in name:
        #         skip = True
        #         break
        if "Spongy" not in name:
            print(f"Skipping {name}")
            continue
        flat_item = flat_dir / name
        mapping[str(res_item)] = str(flat_item)
    for f, gt in mapping.items():
        print(f"processing {f} with {gt}")
        process_all_from_folders(f, gt)
    # gog("./fin/d/2/", 40)

    # gog("./fin/d/3/", 40)

    # gog("./fin/d/4/", 40)

    # gog("./fin/res/", 40)
    # gog("./masks/44logits/", 5)
