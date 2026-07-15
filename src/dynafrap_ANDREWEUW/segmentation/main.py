import os
import sys

import torchio.types as torchio_types

from dynafrap_andreweuw.paths import PRED_DIR_IN, PRED_DIR_OUT_BASE, UNET_LOG_DIR

sys.modules.setdefault("torchio.typing", torchio_types)

from biom3d.builder import Builder  # noqa: E402


def predict_tree(log, dir_in_base, dir_out_base):
    """Run prediction for every directory that directly contains .tiff files.

    The output is written into a mirror of the input folder structure rooted
    at ``dir_out_base``.
    """
    builder = Builder(config=None, path=log, training=False)
    for root, _dirs, files in os.walk(dir_in_base):
        if not any(f.lower().endswith(".tif") for f in files):
            continue
        rel = os.path.relpath(root, dir_in_base)
        dir_out = dir_out_base if rel == os.curdir else os.path.join(dir_out_base, rel)
        print(f"Predicting: {root} -> {dir_out}")
        builder.run_prediction_folder(dir_in=str(root), dir_out= str(dir_out), return_logit=True)

def main() -> None:
    """Run prediction for every directory that directly contains .tiff files.

    The output is written into a mirror of the input folder structure rooted
    at ``dir_out_base``.
    """
    predict_tree(log=UNET_LOG_DIR, dir_in_base=PRED_DIR_IN, dir_out_base=PRED_DIR_OUT_BASE)
