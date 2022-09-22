# From https://github.com/lxtGH/PFSegNets/blob/master/tools/split_iSAID.py
import cv2
import os
import numpy as np
from natsort import natsorted
from glob import glob
from shutil import copyfile
import argparse
from multiprocessing.pool import ThreadPool
from PIL import Image

# BGR
mask_mapping = {
    (0, 0, 0): 0,
    (63, 0, 0): 1,
    (63, 63, 0): 2,
    (0, 63, 0): 3,
    (127, 63, 0): 4,
    (191, 63, 0): 5,
    (255, 63, 0): 6,
    (63, 127, 0): 7,
    (127, 127, 0): 8,
    (127, 0, 0): 9,
    (191, 0, 0): 10,
    (255, 0, 0): 11,
    (127, 191, 0): 12,
    (191, 127, 0): 13,
    (255, 127, 0): 14,
    (155, 100, 0): 15
}

pool = ThreadPool(16)


def parse_args():
    parser = argparse.ArgumentParser(description='Crop iSAID Dataset')
    parser.add_argument('--src',
                        default='./datasets/iSAID',
                        type=str,
                        help='path for the original dataset')
    parser.add_argument('--target',
                        default='./datasets/iSAID_Patches',
                        type=str,
                        help='path for saving the new dataset')
    parser.add_argument('--patch_width',
                        default=800,
                        type=int,
                        help='Width of the cropped image patch')
    parser.add_argument('--patch_height',
                        default=800,
                        type=int,
                        help='Height of the cropped image patch')
    parser.add_argument('--overlap_area',
                        default=200,
                        type=int,
                        help='Overlap area')
    args = parser.parse_args()
    return args


def slide_crop(file_, src_path, target_path, patch_H, patch_W, overlap, extra):
    if file_ == 'P1527' or file_ == 'P1530':
        return 0
    filename = file_ + extra + '.png'
    full_filename = src_path + '/' + filename
    img = cv2.imread(full_filename)
    img_H, img_W, _ = img.shape
    if extra == '_instance_color_RGB':
        mask_gray = np.zeros(img.shape[:2])
        for k, v in mask_mapping.items():
            mask_gray[(img == k).all(axis=2)] = v
        img = mask_gray

    if img_H > patch_H and img_W > patch_W:
        for x in range(0, img_W, patch_W - overlap):
            for y in range(0, img_H, patch_H - overlap):
                x_str = x
                x_end = x + patch_W
                if x_end > img_W:
                    diff_x = x_end - img_W
                    x_str -= diff_x
                    x_end = img_W
                y_str = y
                y_end = y + patch_H
                if y_end > img_H:
                    diff_y = y_end - img_H
                    y_str -= diff_y
                    y_end = img_H
                if extra == '_instance_color_RGB':
                    patch = img[y_str:y_end, x_str:x_end]
                else:
                    patch = img[y_str:y_end, x_str:x_end, :]
                image = file_ + '_' + str(y_str) + '_' + str(
                    y_end) + '_' + str(x_str) + '_' + str(
                        x_end) + extra + '.png'
                save_path_image = target_path + '/' + image
                if extra == '_instance_color_RGB':
                    mask_gray = Image.fromarray(patch.astype('uint8')).convert('L')
                    mask_gray.save(save_path_image)
                else:
                    cv2.imwrite(save_path_image, patch)
    else:
        if extra == '_instance_color_RGB':
            mask_gray = Image.fromarray(img.astype('uint8')).convert('L')
            mask_gray.save(target_path + '/' + filename)
        else:
            copyfile(full_filename, target_path + '/' + filename)


def main():
    args = parse_args()
    src = args.src
    target = args.target
    sub_set = ['train', 'val', 'test']
    image_sub_folder = ['images', 'Semantic_masks']
    patch_H, patch_W = args.patch_width, args.patch_height  # image patch width and height
    overlap = args.overlap_area  # overlap area

    for set in sub_set:
        for folder in image_sub_folder:
            if set == 'test' and folder == 'Semantic_masks':
                continue
            if folder == 'Semantic_masks':
                extra = '_instance_color_RGB'
            else:
                extra = ''

            src_path = src + "/" + set + "/" + folder
            target_path = target + "/" + set + "/" + folder
            os.makedirs(target_path, exist_ok=True)

            files = glob(src_path + "/*.png")
            files_tmp = []
            for i in files:
                file = os.path.split(i)[-1]
                if '_' in file:
                    files_tmp.append(file.split('_')[0])
                else:
                    files_tmp.append(file.split('.')[0])
            files = natsorted(files_tmp)
            for file_ in files:
                pool.apply_async(slide_crop, (
                    file_,
                    src_path,
                    target_path,
                    patch_H,
                    patch_W,
                    overlap,
                    extra,
                ))
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
