import warnings

from jseg.utils.registry import TRANSFORMS
from jseg.utils.general import is_list_of
from .compose import Compose


@TRANSFORMS.register_module()
class MultiScaleFlipAug(object):
    def __init__(self,
                 transforms,
                 img_scale,
                 img_ratios=None,
                 flip=False,
                 flip_direction='horizontal'):
        self.transforms = Compose(transforms)
        if img_ratios is not None:
            img_ratios = img_ratios if isinstance(img_ratios,
                                                  list) else [img_ratios]
            assert is_list_of(img_ratios, float)
        if img_scale is None:
            # mode 1: given img_scale=None and a range of image ratio
            self.img_scale = None
            assert is_list_of(img_ratios, float)
        elif isinstance(img_scale, tuple) and is_list_of(img_ratios, float):
            assert len(img_scale) == 2
            # mode 2: given a scale and a range of image ratio
            self.img_scale = [(int(img_scale[0] * ratio),
                               int(img_scale[1] * ratio))
                              for ratio in img_ratios]
        else:
            # mode 3: given multiple scales
            self.img_scale = img_scale if isinstance(img_scale,
                                                     list) else [img_scale]
        assert is_list_of(self.img_scale, tuple) or self.img_scale is None
        self.flip = flip
        self.img_ratios = img_ratios
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip
                and not any([t['type'] == 'RandomFlip' for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

    def __call__(self, results):
        aug_data = []
        if self.img_scale is None and is_list_of(self.img_ratios, float):
            h, w = results['img'].shape[:2]
            img_scale = [(int(w * ratio), int(h * ratio))
                         for ratio in self.img_ratios]
        else:
            img_scale = self.img_scale
        flip_aug = [False, True] if self.flip else [False]
        for scale in img_scale:
            for flip in flip_aug:
                for direction in self.flip_direction:
                    _results = results.copy()
                    _results['scale'] = scale
                    _results['flip'] = flip
                    _results['flip_direction'] = direction
                    data = self.transforms(_results)
                    aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip})'
        repr_str += f'flip_direction={self.flip_direction}'
        return repr_str
