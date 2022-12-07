# JSeg

**09/27/2022**

**[ External Attention ](python/jseg/ops/external_attention.py) have been accepted by TPAMI.**


**09/19/2022**

**[SegNeXt ](project/segnext) have been accepted by NeurIPS'2022.**


## Introduction
JSeg is a Semantic segmentation toolbox based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [Jittor](https://github.com/Jittor/jittor) and [JDet](https://github.com/Jittor/JDet)

<!-- **Features**
- Automatic compilation. Our framwork is based on Jittor, which means we don't need to Manual compilation for these code with CUDA and C++. -->

<!-- Framework details are avaliable in the [framework.md](docs/framework.md) -->

## Install
JSeg environment requirements:

* System: **Linux**(e.g. Ubuntu/CentOS/Arch), **macOS**, or **Windows Subsystem of Linux (WSL)**
* Python version >= 3.7
* CPU compiler (require at least one of the following)
    * g++ (>=5.4.0)
    * clang (>=8.0)
* GPU compiler (optional)
    * nvcc (>=10.0 for g++ or >=10.2 for clang)
* GPU library: cudnn-dev (recommend tar file installation, [reference link](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar))

**Step 1: Install the requirements**
```shell
git clone https://github.com/Jittor/JSeg
cd JSeg
python -m pip install -r requirements.txt
```
If you have any installation problems for Jittor, please refer to [Jittor](https://github.com/Jittor/jittor)

**Step 2: Install JSeg**
 
```shell
cd JSeg
# suggest this 
python setup.py develop
# or
python setup.py install
```
If you don't have permission for install,please add ```--user```.

## Getting Started

### Datasets
Please check the [dataset_prepare](docs/dataset_prepare.md) for dataset preparation, The preparation of the dataset comes from [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md).

### Config
JSeg defines the used model, dataset and training/testing method by `config-file`, please check the [config.md](docs/config.md) to learn how it works.

### Train
We support single-machine single-gpu, single-machine multi-gpu training, multi-machine training is not supported for the time being. Multi-gpu dependence can be referred to [here](https://cg.cs.tsinghua.edu.cn/jittor/tutorial/2020-5-2-16-44-distributed/)
```shell
python tools/run_net.py --config-file=path/to/config --task=train

# For example
# Single GPU
python tools/run_net.py --config-file=project/fcn/fcn_r50-d8_512x1024_cityscapes_80k.py --task=train

# Multiple GPUs
mpirun -n 8 python tools/run_net.py --config-file=project/fcn/fcn_r50-d8_512x1024_cityscapes_80k.py --task=train
```

### Val
We provide an evaluation script to evaluate the dataset. If there is not enough CPU memory, you can save CPU memory by setting ```--efficient_val``` to store the evaluation results in a local file.
```shell
python tools/run_net.py --config-file=path/to/config --resume=path/to/ckp --task=val

# For example
python tools/run_net.py --config-file=project/fcn/fcn_r50-d8_512x1024_cityscapes_80k.py --resume=work_dirs/fcn_r50-d8_512x1024_cityscapes_80k/checkpoints/ckpt_80000.pkl --task=val
```

### Test for save result
We provide a test scripts to save the inference results of the data set, which can be saved in the specified location by setting ```--save-dir```.
```shell
python tools/run_net.py --config-file=path/to/config --resume=path/to/ckp --save-dir=path/to/save_dir --task=test

# For example
python tools/run_net.py --config-file=project/fcn/fcn_r50-d8_512x1024_cityscapes_80k.py --resume=work_dirs/fcn_r50-d8_512x1024_cityscapes_80k/checkpoints/ckpt_80000.pkl --save-dir=./ --task=test
```

### Demo
We provide a demo that can predict a single picture. For more information, please see [here](tools/demo.py)

```python
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

```

### Supported backbones:
- :heavy_check_mark: ResNet (CVPR'2016)
- :heavy_check_mark: ResNeXt (CVPR'2017)
- :heavy_check_mark: [ResNeSt (ArXiv'2020)](project/resnest)
- :heavy_check_mark: [Vision Transformer (ICLR'2021)](project/vit)
- :heavy_check_mark: [Swin Transformer (ICCV'2021)](project/swin)
- :heavy_check_mark: [ConvNeXt (CVPR'2022)](project/convnext)
- :heavy_check_mark: [BEiT (ICLR'2022)](project/beit)
- :heavy_check_mark: [MAE (CVPR'2022)](project/mae)

### Supported methods:
- :heavy_check_mark: [FCN (CVPR'2015/TPAMI'2017)](project/fcn)
- :heavy_check_mark: [PSPNet (CVPR'2017)](project/pspnet)
- :heavy_check_mark: [DeepLabV3 (ArXiv'2017)](project/deeplabv3)
- :heavy_check_mark: [DeepLabV3+ (CVPR'2018)](project/deeplabv3plus)
- :heavy_check_mark: [UPerNet (ECCV'2018)](project/upernet)
- :heavy_check_mark: [DANet (CVPR'2019)](project/danet)
- :heavy_check_mark: [CCNet (ICCV'2019)](project/ccnet)
- :heavy_check_mark: [PointRend (CVPR'2020)](project/point_rend)
- :heavy_check_mark: [SegFormer (NeurIPS'2021)](project/segformer)
- :heavy_check_mark: [EANet (TPAMI)](project/eanet)
- :heavy_check_mark: [SegNeXt (NeurIPS'2022)](project/segnext)

### Supported datasets:
  - :heavy_check_mark: [ADE20K](docs/dataset_prepare.md#ade20k)
  - :heavy_check_mark: [Cityscapes](docs/dataset_prepare.md#cityscapes)
  - :heavy_check_mark: [PASCAL VOC](docs/dataset_prepare.md#pascal-voc)
  - :heavy_check_mark: [iSAID](docs/dataset_prepare.md#isaid)
  - :heavy_check_mark: [LoveDA](docs/dataset_prepare.md#LoveDA)
  - :heavy_check_mark: [Potsdam](docs/dataset_prepare.md#isprs-potsdam)
  - :heavy_check_mark: [Vaihingen](docs/dataset_prepare.md#isprs-vaihingen)


## Contact Us
Website: http://cg.cs.tsinghua.edu.cn/jittor/

Email: jittor@qq.com

File an issue: https://github.com/Jittor/jittor/issues

QQ Group: 761222083

<img src="https://cg.cs.tsinghua.edu.cn/jittor/images/news/2020-12-8-21-19-1_2_2/fig4.png" width="200"/>

## The Team
JSeg is currently maintained by the [Tsinghua CSCG Group](https://cg.cs.tsinghua.edu.cn/). If you are also interested in JSeg and want to improve it, Please join us!


## Citation


```
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}
```

## Reference
1. [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
2. [Jittor](https://github.com/Jittor/jittor)
3. [JDet](https://github.com/Jittor/JDet)
4. [mmcv](https://github.com/open-mmlab/mmcv)
5. [timm](https://github.com/rwightman/pytorch-image-models)


