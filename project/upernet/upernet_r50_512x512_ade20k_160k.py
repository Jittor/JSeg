_base_ = [
    '../_base_/datasets/ade20k.py', '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='EncoderDecoder',
    pretrained='jittorhub://resnet50_v1c-2cccc1ad.pkl',
    backbone=dict(type='ResNetV1c',
                  depth=50,
                  num_stages=4,
                  out_indices=(0, 1, 2, 3),
                  dilations=(1, 1, 1, 1),
                  strides=(1, 2, 2, 2),
                  norm_eval=False,
                  contract_dilation=True),
    decode_head=dict(type='UPerHead',
                     in_channels=[256, 512, 1024, 2048],
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
                        in_channels=1024,
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
    test_cfg=dict(mode='whole'))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)

max_iter = 160000
eval_interval = 8000
checkpoint_interval = 8000

scheduler = dict(type='PolyLR', max_steps=max_iter, power=0.9, min_lr=1e-4)
