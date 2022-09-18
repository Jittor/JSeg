from jseg.utils.inference import InferenceSegmentor


def main():
    config_file = 'project/fcn/fcn_r50-d8_512x1024_cityscapes_80k.py'
    ckp_file = 'work_dirs/fcn_r50-d8_512x1024_cityscapes_80k/checkpoints/ckpt_80000.pkl'
    save_dir = './'
    image = 'cityscapes/leftImg8bit/val/munster/munster_000069_000019_leftImg8bit.png'

    inference_segmentor = InferenceSegmentor(config_file, ckp_file, save_dir)
    inference_segmentor.infer(image)


if __name__ == "__main__":
    main()
