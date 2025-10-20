from loader import *
from process import crop_all, normalize_all, pick_channel
from viewer import VoxelViewer

def process_all_from_folders(folder_path, gt_path):
    # volumes, filenames = load_3d_tiff_sequence(folder_path)
    volumes, images, filenames = load_dual_3d_tiff_sequence(folder_path, gt_path)
    for volume, filename, images in zip(volumes, filenames, images):
        print(volume.shape)
        bbox, _ = compute_global_bounding_box(volume)
        volume = crop_volumes_to_bounding_box(volume, bbox)

        volume = pick_channel(volume, 1)
        volume, min = crop_all(volume, 0.9, 20)
        viewer = VoxelViewer(volume, filename, images, min, spacing=(1, 1, 3))  # adjust voxel spacing if needed
        viewer.start()  

if __name__ == "__main__":
    # Do we know how many one palisade cell can have?
    # can some appear and some others vanish?

    # inspect





    # Generate results for the different light conditions, make cylinders less high
    # freddy reinvestigates the plots 
    # richer output



    # article doesnt have to be before the bigger paper, can be inconporated into the image analysis platform
    




    # all tifs?
    to_process = [
        # "./res/palisade/",
        "./res/pavement/",
        # "./res/spongy/,"
        # './res/guard_1_1/',
        # './res/guard_1_2/,'
        './res/guard_full/'
    ]
    to_process_im = [
        # "./flat/WL-Palisade.lif_flattened/",
        "./flat/WL-Pavement-001.lif_flattened/",
        # "./flat/WL-Spongy.lif_flattened/,"
        # "./flat/WL-Guard-1-1-005.lif_flattened/",
        # "./flat/WL-Guard-1-2.lif_flattened/,"
        './flat/WL-Guard Full Nucleus.lif_flattened/'
    ]
    for f, gt in zip(to_process, to_process_im):
        process_all_from_folders(f, gt)
    # gog("./fin/d/2/", 40)
    
    # gog("./fin/d/3/", 40)
    
    # gog("./fin/d/4/", 40)

    # gog("./fin/res/", 40)
    # gog("./masks/44logits/", 5)
