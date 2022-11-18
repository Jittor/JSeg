import jittor as jt
from jittor import nn

from jseg.ops import ConvModule, resize

from jseg.utils.registry import HEADS
from .decode_head import BaseDecodeHead


class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.
    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
    """

    def __init__(self, dilations, in_channels, channels):
        super(ASPPModule, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        for dilation in dilations:
            self.append(
                ConvModule(self.in_channels,
                           self.channels,
                           1 if dilation == 1 else 3,
                           dilation=dilation,
                           padding=0 if dilation == 1 else dilation))

    def execute(self, x):
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs


@HEADS.register_module()
class ASPPHead(BaseDecodeHead):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.
    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.
    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self, dilations=(1, 6, 12, 18), **kwargs):
        super(ASPPHead, self).__init__(**kwargs)
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(self.in_channels, self.channels, 1))
        self.aspp_modules = ASPPModule(dilations, self.in_channels,
                                       self.channels)
        self.bottleneck = ConvModule((len(dilations) + 1) * self.channels,
                                     self.channels,
                                     3,
                                     padding=1)

    def _execute_feature(self, inputs):
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(self.image_pool(x),
                   size=x.size()[2:],
                   mode='bilinear',
                   align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = jt.concat(aspp_outs, dim=1)
        feats = self.bottleneck(aspp_outs)
        return feats

    def execute(self, inputs):
        """Forward function."""
        output = self._execute_feature(inputs)
        output = self.cls_seg(output)
        return output
