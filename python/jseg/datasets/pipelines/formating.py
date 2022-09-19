import numpy as np

from jseg.utils.registry import TRANSFORMS
import jittor as jt


def to_tensor(data):
    return jt.Var(data)


@TRANSFORMS.register_module()
class DefaultFormatBundle(object):
    def __call__(self, results):

        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = to_tensor(img)
        if 'gt_semantic_seg' in results:
            # convert to long
            results['gt_semantic_seg'] = to_tensor(
                results['gt_semantic_seg'][None, ...].astype(np.int64))
        return results

    def __repr__(self):
        return self.__class__.__name__


@TRANSFORMS.register_module()
class ImageToTensor(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = to_tensor(img.transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@TRANSFORMS.register_module()
class Collect(object):
    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = img_meta
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys}, meta_keys={self.meta_keys})'
