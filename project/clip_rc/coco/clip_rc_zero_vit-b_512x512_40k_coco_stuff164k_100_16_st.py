_base_ = [
    '../datasets/zero_cocostuff_512x512.py', '../../_base_/default_runtime.py'
]

img_size = 512
in_channels = 512
out_indices = [11]

region_level_bridge_size = 16
base_class = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21,
    22, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44,
    45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 83, 84,
    85, 86, 87, 89, 90, 91, 92, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103,
    104, 105, 106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 134, 135,
    138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
    153, 154, 155, 156, 158, 159, 161, 162, 163, 164, 165, 166, 167, 168, 169,
    170
]
novel_class = [
    19, 23, 28, 29, 36, 51, 76, 88, 94, 112, 133, 136, 137, 157, 160
]
both_class = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
    78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
    97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
    127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
    142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156,
    157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170
]
CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
           'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
           'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
           'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
           'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
           'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower',
           'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel',
           'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
           'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper',
           'pavement', 'pillow', 'plant-other', 'plastic', 'platform',
           'playingfield', 'railing', 'railroad', 'river', 'road', 'rock',
           'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other',
           'skyscraper', 'snow', 'solid-other', 'stairs', 'stone', 'straw',
           'structural-other', 'table', 'tent', 'textile-other', 'towel',
           'tree', 'vegetable', 'wall-brick', 'wall-concrete', 'wall-other',
           'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other',
           'waterdrops', 'window-blind', 'window-other', 'wood')

pretrained = 'ViT-B-16.pkl'

exclude_key = ['prompt']

model = dict(
    type='CLIPRC',
    pretrained=pretrained,
    pretrained_text=pretrained,
    class_names=CLASSES,
    backbone=dict(
        type='CLIPVisionTransformerWithRLB',
        patch_size=16,
        width=768,
        output_dim=512,
        get_embeddings=True,
        drop_path_rate=0.1,
        layers=12,
        input_resolution=img_size,
        out_indices=out_indices,
        # setting of vpt
        num_tokens=100,
        prompt_dim=768,
        total_d_layer=11,
        # setting of RLB
        region_level_bridge_size=region_level_bridge_size),
    text_encoder=dict(type='CLIPTextEncoder',
                      context_length=77,
                      embed_dim=512,
                      transformer_width=512,
                      transformer_heads=8,
                      transformer_layers=12),
    decode_head=dict(
        type='ATMSingleHeadSeg',
        img_size=img_size,
        in_channels=in_channels,
        seen_idx=base_class,
        all_idx=both_class,
        channels=in_channels,
        num_layers=3,
        num_classes=len(both_class),  # useless, to decode_head,
        num_heads=8,
        use_proj=False,
        use_stages=len(out_indices),
        embed_dims=in_channels),
    test_cfg=dict(mode='slide',
                  crop_size=(img_size, img_size),
                  stride=(426, 426)),
    base_class=base_class,
    novel_class=novel_class,
    both_class=both_class,
    self_training=True,
    ft_backbone=False,
    exclude_key='prompt',
    load_text_embedding='project/clip_rc/text_embedding/coco_multi.npy')

parameter_groups_generator = dict(type="CustomPrameterGroupsGenerator",
                                  custom_keys={
                                      'backbone': dict(lr_mult=10.0),
                                      'text_encoder': dict(lr_mult=0.0),
                                      'norm': dict(decay_mult=0.),
                                      'ln': dict(decay_mult=0.),
                                      'head': dict(lr_mult=10.),
                                  })

optimizer = dict(
    type='CustomAdamW',
    lr=0.00002,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)

max_iter = 40000
eval_interval = 2000
checkpoint_interval = 2000

scheduler = dict(type='PolyLR',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 max_steps=max_iter,
                 power=0.9,
                 min_lr=1e-6)
