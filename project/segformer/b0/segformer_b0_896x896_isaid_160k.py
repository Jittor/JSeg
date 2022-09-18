# model settings
model = dict(
    type='EncoderDecoder',
    pretrained='jittorhub://mit_b0.pkl',
    backbone=dict(type='mit_b0'),
    decode_head=dict(type='SegFormerHead',
                     in_channels=[32, 64, 160, 256],
                     in_index=[0, 1, 2, 3],
                     feature_strides=[4, 8, 16, 32],
                     channels=128,
                     dropout_ratio=0.1,
                     num_classes=16,
                     align_corners=False,
                     decoder_params=dict(embed_dim=256),
                     loss_decode=dict(type='CrossEntropyLoss',
                                      use_sigmoid=False,
                                      loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset settings
dataset_type = 'iSAIDDataset'
data_root = '/home/gmh/dataset/iSAID_patches'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)

crop_size = (896, 896)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(896, 896), ratio_range=(0.5, 2.0)),
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
        img_scale=(896, 896),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='AlignedResize', keep_ratio=True, size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset = dict(
    train=dict(type=dataset_type,
               batch_size=16,
               num_workers=4,
               shuffle=True,
               drop_last=False,
               data_root=data_root,
               img_dir='train/images',
               ann_dir='train/masks_new',
               pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # Fixed to one
        batch_size=1,
        num_workers=1,
        shuffle=False,
        drop_last=False,
        data_root=data_root,
        img_dir='val/images',
        ann_dir='val/masks_new',
        pipeline=test_pipeline))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)

max_iter = 160000
eval_interval = 8000
checkpoint_interval = 8000
log_interval = 50

scheduler = dict(type='PolyLR',
                 warmup='linear',
                 warmup_iters=1000,
                 warmup_ratio=0.001,
                 max_steps=max_iter,
                 power=0.9,
                 min_lr=1e-4)

logger = dict(type="RunLogger")
