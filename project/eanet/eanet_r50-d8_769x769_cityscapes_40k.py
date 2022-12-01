_base_ = [
    '../_base_/datasets/cityscapes_769x769.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
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
    decode_head=dict(type='EAHead',
                     in_channels=2048,
                     in_index=3,
                     channels=512,
                     dropout_ratio=0.1,
                     num_classes=19,
                     align_corners=True,
                     loss_decode=dict(type='CrossEntropyLoss',
                                      use_sigmoid=False,
                                      loss_weight=1.0,
                                      class_weight=[
                                          0.8373, 0.918, 0.866, 1.0345, 1.0166,
                                          0.9969, 0.9754, 1.0489, 0.8786,
                                          1.0023, 0.9539, 0.9843, 1.1116,
                                          0.9037, 1.0865, 1.0955, 1.0865,
                                          1.1529, 1.0507
                                      ]),
                     sampler=dict(type='OHEMPixelSampler',
                                  thresh=0.7,
                                  min_kept=100000)),
    auxiliary_head=dict(type='FCNHead',
                        in_channels=1024,
                        in_index=2,
                        channels=256,
                        num_convs=1,
                        concat_input=False,
                        dropout_ratio=0.1,
                        num_classes=19,
                        norm_cfg=norm_cfg,
                        align_corners=True,
                        loss_decode=dict(type='CrossEntropyLoss',
                                         use_sigmoid=False,
                                         loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))
