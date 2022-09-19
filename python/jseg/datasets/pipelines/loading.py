import os.path as osp

import numpy as np
from jseg.utils.registry import TRANSFORMS
from PIL import Image
from .utils import _pillow2array


@TRANSFORMS.register_module()
class LoadImageFromFile(object):
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img = Image.open(filename)
        img = _pillow2array(img,
                            flag=self.color_type,
                            channel_order=self.channel_order)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = np.array(img.shape)
        results['ori_shape'] = np.array(img.shape)
        # Set initial values for default meta_keys
        results['pad_shape'] = np.array(img.shape)
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(mean=np.zeros(num_channels,
                                                     dtype=np.float32),
                                       std=np.ones(num_channels,
                                                   dtype=np.float32),
                                       to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@TRANSFORMS.register_module()
class LoadAnnotations(object):
    def __init__(self, reduce_zero_label=False):
        self.reduce_zero_label = reduce_zero_label

    def __call__(self, results):
        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        gt_semantic_seg = Image.open(filename)
        gt_semantic_seg = _pillow2array(
            gt_semantic_seg, flag='unchanged').squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
