import cv2
import os
import numpy as np
import warnings


def visualize_result(seg, palette=None, save_dir=None, file_name=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    out_file = os.path.join(save_dir, file_name[file_name.rfind("/") + 1:])
    if palette is None:
        palette = np.random.randint(0, 255, size=(255, 3))
        warnings.warn('palette is not specified, random palette is used')

    palette = np.array(palette)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]
    color_seg = color_seg.astype(np.uint8)
    cv2.imwrite(out_file, color_seg)
