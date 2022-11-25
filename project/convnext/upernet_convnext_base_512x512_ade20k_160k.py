model = dict(
    type='EncoderDecoder',
    pretrained='jittorhub://convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pkl',
    backbone=dict(type='ConvNeXt',
                  arch='base',
                  out_indices=[0, 1, 2, 3],
                  drop_path_rate=0.4,
                  layer_scale_init_value=1.0,
                  gap_before_final_norm=False),
    decode_head=dict(type='UPerHead',
                     in_channels=[128, 256, 512, 1024],
                     in_index=[0, 1, 2, 3],
                     pool_scales=(1, 2, 3, 6),
                     channels=512,
                     dropout_ratio=0.1,
                     num_classes=150,
                     align_corners=False,
                     loss_decode=dict(type='CrossEntropyLoss',
                                      use_sigmoid=False,
                                      loss_weight=1.0)),
    auxiliary_head=dict(type='FCNHead',
                        in_channels=512,
                        in_index=2,
                        channels=256,
                        num_convs=1,
                        concat_input=False,
                        dropout_ratio=0.1,
                        num_classes=150,
                        align_corners=False,
                        loss_decode=dict(type='CrossEntropyLoss',
                                         use_sigmoid=False,
                                         loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))

dataset_type = 'ADE20KDataset'
data_root = '/home/gmh/datasets/ADEChallengeData2016/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
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
    dict(type='MultiScaleFlipAug',
         img_scale=(2048, 512),
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
               batch_size=16,
               num_workers=8,
               shuffle=True,
               drop_last=False,
               data_root=data_root,
               img_dir='images/training',
               ann_dir='annotations/training',
               pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # Fixed to one
        batch_size=1,
        num_workers=1,
        shuffle=False,
        drop_last=False,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline))

# TODO
parameter_groups_generator = dict(type="CustomPrameterGroupsGenerator",
                                  custom_keys={})

optimizer = dict(
    type='CustomAdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
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
