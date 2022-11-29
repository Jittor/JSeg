_base_ = [
    '../../_base_/datasets/ade20k.py', '../../_base_/default_runtime.py'
]

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
                     num_classes=150,
                     align_corners=False,
                     decoder_params=dict(embed_dim=256),
                     loss_decode=dict(type='CrossEntropyLoss',
                                      use_sigmoid=False,
                                      loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

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
