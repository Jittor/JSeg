_base_ = [
    '../../_base_/datasets/ade20k.py', '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='EncoderDecoder',
    pretrained='jittorhub://swin_tiny_patch4_window7_224_22k.pkl',
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
    ),
    decode_head=dict(type='LightHamHead',
                     in_channels=[192, 384, 768],
                     in_index=[1, 2, 3],
                     channels=256,
                     dropout_ratio=0.1,
                     num_classes=150,
                     align_corners=False,
                     loss_decode=dict(type='CrossEntropyLoss',
                                      use_sigmoid=False,
                                      loss_weight=1.0),
                     ham_channels=256,
                     ham_kwargs=dict(MD_R=16)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

parameter_groups_generator = dict(type="CustomPrameterGroupsGenerator",
                                  custom_keys={
                                      'absolute_pos_embed':
                                      dict(decay_mult=0.),
                                      'relative_position_bias_table':
                                      dict(decay_mult=0.),
                                      'norm':
                                      dict(decay_mult=0.)
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
