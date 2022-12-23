import math
from jittor import nn
from jseg.utils.registry import CONV_LAYERS
from typing import Tuple, Union

CONV_LAYERS.register_module('Conv1d', module=nn.Conv1d)
CONV_LAYERS.register_module('Conv2d', module=nn.Conv2d)
CONV_LAYERS.register_module('Conv3d', module=nn.Conv3d)
CONV_LAYERS.register_module('Conv', module=nn.Conv2d)


class Conv2dAdaptivePadding(nn.Conv2d):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0,
                         dilation, groups, bias)

    def execute(self, x):
        img_h, img_w = x.size()[-2:]
        kernel_h, kernel_w = self.weight.size()[-2:]
        stride_h, stride_w = self.stride
        output_h = math.ceil(img_h / stride_h)
        output_w = math.ceil(img_w / stride_w)
        pad_h = (max((output_h - 1) * self.stride[0] +
                     (kernel_h - 1) * self.dilation[0] + 1 - img_h, 0))
        pad_w = (max((output_w - 1) * self.stride[1] +
                     (kernel_w - 1) * self.dilation[1] + 1 - img_w, 0))
        if pad_h > 0 or pad_w > 0:
            x = nn.pad(x, [
                pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2
            ])
        return nn.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                         self.dilation, self.groups)


CONV_LAYERS.register_module('Conv2dAdaptivePadding',
                            module=Conv2dAdaptivePadding)


def build_conv_layer(cfg, *args, **kwargs):
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    conv_layer = CONV_LAYERS.get(layer_type)

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer
