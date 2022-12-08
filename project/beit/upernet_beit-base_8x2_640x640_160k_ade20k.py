_base_ = [
    '../_base_/datasets/ade20k_640x640.py', '../_base_/default_runtime.py'
]

norm_cfg = dict(type='BN')
model = dict(
    type='EncoderDecoder',
    pretrained='jittorhub://beit_base_patch16_224_pt22k_ft22k.pkl',
    backbone=dict(type='BEiT',
                  img_size=(640, 640),
                  patch_size=16,
                  in_channels=3,
                  embed_dims=768,
                  num_layers=12,
                  num_heads=12,
                  mlp_ratio=4,
                  out_indices=(3, 5, 7, 11),
                  qv_bias=True,
                  attn_drop_rate=0.0,
                  drop_path_rate=0.1,
                  norm_cfg=dict(type='LN', eps=1e-6),
                  act_cfg=dict(type='GELU'),
                  norm_eval=False,
                  init_values=0.1),
    neck=dict(type='Feature2Pyramid', embed_dim=768, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(type='UPerHead',
                     in_channels=[768, 768, 768, 768],
                     in_index=[0, 1, 2, 3],
                     pool_scales=(1, 2, 3, 6),
                     channels=768,
                     dropout_ratio=0.1,
                     num_classes=150,
                     norm_cfg=norm_cfg,
                     align_corners=False,
                     loss_decode=dict(type='CrossEntropyLoss',
                                      use_sigmoid=False,
                                      loss_weight=1.0)),
    auxiliary_head=dict(type='FCNHead',
                        in_channels=768,
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
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(426, 426)))

parameter_groups_generator = dict(type="LRDecayParameterGroupsGenerator",
                                  paramwise_cfg=dict(num_layers=12,
                                                     decay_rate=0.9))

optimizer = dict(
    type='CustomAdamW',
    lr=3e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
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
