## Prepare datasets
It is recommended that you symlink the dataset to `JSeg/dataset`.

### ADE20K
The ADE20K training and validation set can be downloaded from here [link](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip).
You can download the test set from [here](http://data.csail.mit.edu/places/ADEchallenge/release_test.zip).
```
├── ADEChallengeData2016
│   ├── annotations
│   │   ├── training
│   │   ├── validation
│   ├── images
│   │   ├── training
│   │   ├── validation
```

### Cityscapes

The data could be found [here](https://www.cityscapes-dataset.com/downloads/) after registration.

By convention, `**labelTrainIds.png` are used for cityscapes training.
We provided a [scripts](https://github.com/open-mmlab/mmsegmentation/blob/master/tools/convert_datasets/cityscapes.py) based on [cityscapesscripts](https://github.com/mcordts/cityscapesScripts)
to generate `**labelTrainIds.png`.

```shell
python tools/convert_datasets/cityscapes.py dataset/cityscapes
```

```
├── cityscapes
│   ├── leftImg8bit
│   │   ├── train
│   │   ├── val
│   ├── gtFine
│   │   ├── train
│   │   ├── val
```


### Pascal VOC

Pascal VOC 2012 could be downloaded from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar).
Beside, most recent works on Pascal VOC dataset usually exploit extra augmentation data, which could be found [here](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz).

If you would like to use augmented VOC dataset, please run following command to convert augmentation annotations into proper format.

```shell
python tools/convert_datasets/voc_aug.py data/VOCdevkit data/VOCdevkit/VOCaug
```


```
├── VOCdevkit
│   ├── VOC2012
│   │   ├── JPEGImages
│   │   ├── SegmentationClass
│   │   ├── ImageSets
│   │   │   ├── Segmentation
│   ├── VOC2010
│   │   ├── JPEGImages
│   │   ├── SegmentationClassContext
│   │   ├── ImageSets
│   │   │   ├── SegmentationContext
│   │   │   │   ├── train.txt
│   │   │   │   ├── val.txt
│   │   ├── trainval_merged.json
│   ├── VOCaug
│   │   ├── dataset
│   │   │   ├── cls
```

### iSAID

The data images could be download from [DOTA-v1.0](https://captain-whu.github.io/DOTA/dataset.html) (train/val/test)

The data annotations could be download from [iSAID](https://captain-whu.github.io/iSAID/dataset.html) (train/val)

The dataset is a Large-scale Dataset for Instance Segmentation (also have segmantic segmentation) in Aerial Images.

Original file structure
```
├── iSAID
│   ├── train
│   │   ├── images
│   │   │   ├── part1.zip
│   │   │   ├── part2.zip
│   │   │   ├── part3.zip
│   │   ├── Semantic_masks
│   │   │   ├── images.zip
│   ├── val
│   │   ├── images
│   │   │   ├── part1.zip
│   │   ├── Semantic_masks
│   │   │   ├── images.zip
│   ├── test
│   │   ├── images
│   │   │   ├── part1.zip
│   │   │   ├── part2.zip
```

After decompression
```
├── iSAID
│   ├── train
│   │   ├── images
│   │   │   ├── *.png
│   │   ├── Semantic_masks
│   │   │   ├── *_instance_color_RGB.png
│   ├── val
│   │   ├── images
│   │   │   ├── *.png
│   │   ├── Semantic_masks
│   │   │   ├── *_instance_color_RGB.png
│   ├── test
│   │   ├── images
│   │   │   ├── *.png
```

Processed file structure
```
├── iSAID_Patches
│   ├── train
│   │   ├── images
│   │   │   ├── *_sub_img.png
│   │   ├── Semantic_masks
│   │   │   ├── *_sub_img_instance_color_RGB.png
│   ├── val
│   │   ├── images
│   │   │   ├── *_sub_img.png
│   │   ├── Semantic_masks
│   │   │   ├── *_sub_img_instance_color_RGB.png
│   ├── test
│   │   ├── images
│   │   │   ├── *_sub_img.png
```

```shell
python tools/convert_datasets/isaid.py --src=./datasets/iSAID --target=./datasets/iSAID_Patches
```

In our default setting (`patch_width`=800, `patch_height`=800,　`overlap_area`=200), it will generate 28029 images for training and 9512 images for validation.

### LoveDA

The data could be downloaded from Google Drive [here](https://drive.google.com/drive/folders/1ibYV0qwn4yuuh068Rnc-w4tPi0U0c-ti?usp=sharing).

Or it can be downloaded from [zenodo](https://zenodo.org/record/5706578#.YZvN7SYRXdF), you should run the following command:

```shell
# Download Train.zip
wget https://zenodo.org/record/5706578/files/Train.zip
# Download Val.zip
wget https://zenodo.org/record/5706578/files/Val.zip
# Download Test.zip
wget https://zenodo.org/record/5706578/files/Test.zip
```

For LoveDA dataset, please run the following command to download and re-organize the dataset.

```shell
python tools/convert_datasets/loveda.py /path/to/loveDA
```

More details about LoveDA can be found [here](https://github.com/Junjue-Wang/LoveDA).

### ISPRS Potsdam

The [Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)
dataset is for urban semantic segmentation used in the 2D Semantic Labeling Contest - Potsdam.

For Potsdam dataset, please run the following command to download and re-organize the dataset.

```shell
python tools/convert_datasets/potsdam.py /path/to/potsdam
```

In our default setting, it will generate 3456 images for training and 2016 images for validation.

### ISPRS Vaihingen

The [Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)
dataset is for urban semantic segmentation used in the 2D Semantic Labeling Contest - Vaihingen.

For Vaihingen dataset, please run the following command to download and re-organize the dataset.

```shell
python tools/convert_datasets/vaihingen.py /path/to/vaihingen
```

In our default setting (`clip_size` =512, `stride_size`=256), it will generate 344 images for training and 398 images for validation.