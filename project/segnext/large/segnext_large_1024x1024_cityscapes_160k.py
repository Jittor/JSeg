_base_ = [
    '../../_base_/datasets/cityscapes_1024x1024.py',
    '../../_base_/default_runtime.py',
]

# model settings
norm_cfg = dict(type='GN', num_groups=32)
model = dict(
    type='EncoderDecoder',
    pretrained='jittorhub://mscan_l.pkl',
    backbone=dict(type='MSCAN',
                  embed_dims=[64, 128, 320, 512],
                  mlp_ratios=[8, 8, 4, 4],
                  drop_rate=0.0,
                  drop_path_rate=0.3,
                  depths=[3, 5, 27, 3]),
    decode_head=dict(type='LightHamHead',
                     in_channels=[128, 320, 512],
                     in_index=[1, 2, 3],
                     channels=1024,
                     dropout_ratio=0.1,
                     num_classes=19,
                     norm_cfg=norm_cfg,
                     align_corners=False,
                     loss_decode=dict(type='CrossEntropyLoss',
                                      use_sigmoid=False,
                                      loss_weight=1.0),
                     ham_channels=1024),
    # model training and testing settings
    train_cfg=dict(),
    # test_cfg=dict(mode='whole'))
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
dataset = dict(
    val=dict(pipeline=test_pipeline))

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

scheduler = dict(type='PolyLR',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 max_steps=max_iter,
                 power=1.0,
                 min_lr=0)
