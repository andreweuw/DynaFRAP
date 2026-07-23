from pathlib import Path

from dynafrap_andreweuw.measuring.interactive_viewer.voxel_viewer import VoxelViewer
from dynafrap_andreweuw.measuring.loader import (
    load_dual_3d_tiff_sequence,
)
from dynafrap_andreweuw.measuring.process import crop_all
from dynafrap_andreweuw.paths import MEASUREMENTS_INPUT_FOLDER, SEGMENTATIONS_FOLDER


def process_all_from_folders(folder_path, gt_path, pb1_max_frame=None):
    all_volumes, all_images, all_filenames = load_dual_3d_tiff_sequence(
        folder_path,
        gt_path,
        pb1_max_frame=pb1_max_frame,
    )
    for volumes, filenames, images in zip(all_volumes, all_filenames, all_images, strict=False):
        print(volumes.shape)

        # bbox, _ = compute_global_bounding_box(volumes)
        # volumes = crop_volumes_to_bounding_box(volumes, bbox)
        volumes = volumes[:, 1] # pick channel 1
        volumes, min = crop_all(volumes, 0.99, 20)
        viewer = VoxelViewer(
            volumes, filenames, images, min, spacing=(1, 1, 3)
        )  # adjust voxel spacing if needed
        viewer.start()


def main():
    mapping = {}
    for sample_segmentation_dir in SEGMENTATIONS_FOLDER.iterdir():
        name = sample_segmentation_dir.name
        flat_item = MEASUREMENTS_INPUT_FOLDER / name
        mapping[str(sample_segmentation_dir)] = str(flat_item)
    for f, gt in mapping.items():
        print(f"processing {f} with {gt}")
        process_all_from_folders(f, gt)


if __name__ == "__main__":
    main()
