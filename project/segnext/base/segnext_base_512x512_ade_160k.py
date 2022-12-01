_base_ = [
    '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py',
]

# model settings
norm_cfg = dict(type='GN', num_groups=32)
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
                     num_classes=150,
                     norm_cfg=norm_cfg,
                     align_corners=False,
                     loss_decode=dict(type='CrossEntropyLoss',
                                      use_sigmoid=False,
                                      loss_weight=1.0),
                     ham_channels=512),
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
