_base_ = './zero_voc12_20_512x512.py'
# dataset settings, merge voc12 and voc12aug
dataset = dict(train=dict(ann_dir='SegmentationClassAug',
                          split='ImageSets/Segmentation/trainaug.txt')) # merge voc12 and voc12aug
