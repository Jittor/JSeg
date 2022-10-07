# model settings
model = dict(
    type='EncoderDecoder',
    pretrained='jittorhub://resnet50_v1c-2cccc1ad.pkl',
    backbone=dict(type='ResNetV1c',
                  depth=50,
                  num_stages=4,
                  out_indices=(0, 1, 2, 3),
                  dilations=(1, 1, 2, 4),
                  strides=(1, 2, 1, 1),
                  norm_eval=False,
                  contract_dilation=True),
    decode_head=dict(type='EAHead',
                     in_channels=2048,
                     in_index=3,
                     channels=512,
                     dropout_ratio=0.1,
                     num_classes=19,
                     align_corners=True,
                     loss_decode=dict(type='CrossEntropyLoss',
                                      use_sigmoid=False,
                                      loss_weight=1.0)),
    auxiliary_head=dict(type='FCNHead',
                        in_channels=1024,
                        in_index=2,
                        channels=256,
                        num_convs=1,
                        concat_input=False,
                        dropout_ratio=0.1,
                        num_classes=19,
                        align_corners=True,
                        loss_decode=dict(type='CrossEntropyLoss',
                                         use_sigmoid=False,
                                         loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))

dataset_type = 'CityscapesDataset'
data_root = 'datasets/cityscapes/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
crop_size = (769, 769)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2049, 1025), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1025),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
dataset = dict(
    train=dict(type=dataset_type,
               batch_size=8,
               num_workers=8,
               shuffle=True,
               drop_last=False,
               data_root=data_root,
               img_dir='leftImg8bit/train',
               ann_dir='gtFine/train',
               pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # Fixed to one
        batch_size=1,
        num_workers=1,
        shuffle=False,
        drop_last=False,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # Fixed to one
        batch_size=1,
        num_workers=1,
        shuffle=False,
        drop_last=False,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)

max_iter = 40000
eval_interval = 4000
checkpoint_interval = 4000
log_interval = 50

scheduler = dict(type='PolyLR',
                 max_steps=max_iter,
                 power=0.9,
                 min_lr=1e-4)

logger = dict(type="RunLogger")
