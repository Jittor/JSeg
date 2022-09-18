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

```shell
python tools/convert_datasets/isaid.py /path/to/iSAID
```

In our default setting (`patch_width`=896, `patch_height`=896,　`overlap_area`=384), it will generate 33978 images for training and 11644 images for validation.
