_base_ = [
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='EncoderDecoder',
    pretrained=
    'jittorhub://convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pkl',
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

parameter_groups_generator = dict(type="LRDecayParameterGroupsGenerator",
                                  paramwise_cfg={
                                      'decay_rate': 0.9,
                                      'decay_type': 'stage_wise',
                                      'num_layers': 12
                                  })

optimizer = dict(
    type='CustomAdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
)

max_iter = 160000
scheduler = dict(type='PolyLR',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 max_steps=max_iter,
                 power=1.0,
                 min_lr=0)
