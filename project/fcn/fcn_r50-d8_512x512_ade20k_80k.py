_base_ = [
    '../_base_/datasets/ade20k.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

# model settings
norm_cfg = dict(type='BN')
model = dict(
    type='EncoderDecoder',
    pretrained='jittorhub://resnet50_v1c-2cccc1ad.pkl',
    backbone=dict(type='ResNetV1c',
                  depth=50,
                  num_stages=4,
                  out_indices=(0, 1, 2, 3),
                  dilations=(1, 1, 2, 4),
                  strides=(1, 2, 1, 1),
                  norm_cfg=norm_cfg,
                  norm_eval=False,
                  contract_dilation=True),
    decode_head=dict(type='FCNHead',
                     in_channels=2048,
                     in_index=3,
                     channels=512,
                     num_convs=2,
                     concat_input=True,
                     dropout_ratio=0.1,
                     num_classes=150,
                     norm_cfg=norm_cfg,
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
                        norm_cfg=norm_cfg,
                        align_corners=False,
                        loss_decode=dict(type='CrossEntropyLoss',
                                         use_sigmoid=False,
                                         loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
