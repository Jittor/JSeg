import argparse
import os.path as osp
from jseg.utils.general import mkdir_or_exist, scandir
from cityscapesscripts.preparation.json2labelImg import json2labelImg
from tqdm import tqdm


def convert_json_to_label(json_file):
    label_file = json_file.replace('_polygons.json', '_labelTrainIds.png')
    json2labelImg(json_file, label_file, 'trainIds')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to TrainIds')
    parser.add_argument('cityscapes_path', help='cityscapes data path')
    parser.add_argument('--gt-dir', default='gtFine', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cityscapes_path = args.cityscapes_path
    out_dir = args.out_dir if args.out_dir else cityscapes_path
    mkdir_or_exist(out_dir)

    gt_dir = osp.join(cityscapes_path, args.gt_dir)

    for poly in tqdm(scandir(gt_dir, '_polygons.json', recursive=True)):
        poly_file = osp.join(gt_dir, poly)
        convert_json_to_label(poly_file)

    split_names = ['train', 'val', 'test']

    for split in split_names:
        filenames = []
        for poly in scandir(osp.join(gt_dir, split),
                            '_polygons.json',
                            recursive=True):
            filenames.append(poly.replace('_gtFine_polygons.json', ''))
        with open(osp.join(out_dir, f'{split}.txt'), 'w') as f:
            f.writelines(f + '\n' for f in filenames)


if __name__ == '__main__':
    main()
