import math
from jseg.utils.registry import BACKBONES
from .resnet import ResLayer
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet
from jittor import nn


class Bottleneck(_Bottleneck):

    def __init__(self,
                 inplanes,
                 planes,
                 groups=1,
                 base_width=4,
                 base_channels=64,
                 **kwargs):
        super(Bottleneck, self).__init__(inplanes, planes, **kwargs)

        if groups == 1:
            width = self.planes
        else:
            width = math.floor(self.planes *
                               (base_width / base_channels)) * groups

        self.bn1 = nn.BatchNorm2d(width)
        self.bn2 = nn.BatchNorm2d(width)
        self.bn3 = nn.BatchNorm2d(self.planes * self.expansion)

        self.conv1 = nn.Conv2d(self.inplanes,
                               width,
                               kernel_size=1,
                               stride=self.conv1_stride,
                               bias=False)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = self.dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(width,
                                   width,
                                   kernel_size=3,
                                   stride=self.conv2_stride,
                                   padding=self.dilation,
                                   dilation=self.dilation,
                                   groups=groups,
                                   bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = nn.Conv2d(width,
                                   width,
                                   kernel_size=3,
                                   stride=self.conv2_stride,
                                   padding=self.dilation,
                                   dilation=self.dilation,
                                   groups=groups,
                                   bias=False)

        self.conv3 = nn.Conv2d(width,
                               planes * self.expansion,
                               kernel_size=1,
                               bias=False)


@BACKBONES.register_module()
class ResNeXt(ResNet):
    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, groups=1, base_width=4, **kwargs):
        self.groups = groups
        self.base_width = base_width
        super(ResNeXt, self).__init__(**kwargs)

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``"""
        return ResLayer(groups=self.groups,
                        base_width=self.base_width,
                        base_channels=self.base_channels,
                        **kwargs)
