import math
from jseg.utils.registry import BACKBONES
from .resnet import ResLayer
from jseg.bricks import build_conv_layer, build_norm_layer
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet


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

        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg,
                                                  width,
                                                  postfix=1)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg,
                                                  width,
                                                  postfix=2)
        self.norm3_name, norm3 = build_norm_layer(self.norm_cfg,
                                                  self.planes * self.expansion,
                                                  postfix=3)

        self.conv1 = build_conv_layer(self.conv_cfg,
                                      self.inplanes,
                                      width,
                                      kernel_size=1,
                                      stride=self.conv1_stride,
                                      bias=False)
        setattr(self, self.norm1_name, norm1)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = self.dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(self.conv_cfg,
                                          width,
                                          width,
                                          kernel_size=3,
                                          stride=self.conv2_stride,
                                          padding=self.dilation,
                                          dilation=self.dilation,
                                          groups=groups,
                                          bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(self.dcn,
                                          width,
                                          width,
                                          kernel_size=3,
                                          stride=self.conv2_stride,
                                          padding=self.dilation,
                                          dilation=self.dilation,
                                          groups=groups,
                                          bias=False)

        setattr(self, self.norm2_name, norm2)
        self.conv3 = build_conv_layer(self.conv_cfg,
                                      width,
                                      self.planes * self.expansion,
                                      kernel_size=1,
                                      bias=False)
        setattr(self, self.norm3_name, norm3)


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
