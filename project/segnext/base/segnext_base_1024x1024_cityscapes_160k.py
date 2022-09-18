# model settings
model = dict(
    type='EncoderDecoder',
    pretrained='jittorhub://mscan_b.pkl',
    backbone=dict(type='MSCAN',
                  embed_dims=[64, 128, 320, 512],
                  mlp_ratios=[8, 8, 4, 4],
                  drop_rate=0.0,
                  drop_path_rate=0.2,
                  depths=[3, 3, 12, 3]),
    decode_head=dict(type='LightHamHead',
                     in_channels=[128, 320, 512],
                     in_index=[1, 2, 3],
                     channels=512,
                     dropout_ratio=0.1,
                     num_classes=19,
                     align_corners=False,
                     decoder_params=dict(),
                     loss_decode=dict(type='CrossEntropyLoss',
                                      use_sigmoid=False,
                                      loss_weight=1.0),
                     ham_channels=512),
    # model training and testing settings
    train_cfg=dict(),
    # test_cfg=dict(mode='whole'))
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

dataset_type = 'CityscapesDataset'
data_root = 'datasets/cityscapes/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
crop_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
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
        img_scale=(2048, 1024),
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

parameter_groups_generator = dict(type="CustomPrameterGroupsGenerator",
                                  custom_keys={
                                      'pos_block': dict(decay_mult=0.),
                                      'norm': dict(decay_mult=0.),
                                      'head': dict(lr_mult=10.)
                                  })

optimizer = dict(
    type='CustomAdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)

max_iter = 160000
eval_interval = 8000
checkpoint_interval = 8000
log_interval = 50

scheduler = dict(type='PolyLR',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 max_steps=max_iter,
                 power=1.0,
                 min_lr=0)

logger = dict(type="RunLogger")
