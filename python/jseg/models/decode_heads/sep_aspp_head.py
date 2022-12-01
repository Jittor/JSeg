import jittor as jt
from jittor import nn
from jseg.ops import resize
from jseg.bricks import ConvModule, DepthwiseSeparableConvModule
from jseg.utils.registry import HEADS
from .aspp_head import ASPPHead


class DepthwiseSeparableASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv."""

    def __init__(self, dilations, in_channels, channels, norm_cfg, act_cfg):
        super(DepthwiseSeparableASPPModule, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for i, dilation in enumerate(self.dilations):
            if dilation == 1:
                self.append(
                    ConvModule(self.in_channels,
                               self.channels,
                               1,
                               dilation=dilation,
                               padding=0,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg))
            if dilation > 1:
                self.append(
                    DepthwiseSeparableConvModule(self.in_channels,
                                                 self.channels,
                                                 3,
                                                 dilation=dilation,
                                                 padding=dilation,
                                                 norm_cfg=self.norm_cfg,
                                                 act_cfg=self.act_cfg))

    def execute(self, x):
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))
        return aspp_outs


@HEADS.register_module()
class DepthwiseSeparableASPPHead(ASPPHead):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.
    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.
    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, c1_in_channels, c1_channels, **kwargs):
        super(DepthwiseSeparableASPPHead, self).__init__(**kwargs)
        assert c1_in_channels >= 0
        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(c1_in_channels,
                                            c1_channels,
                                            1,
                                            conv_cfg=self.conv_cfg,
                                            norm_cfg=self.norm_cfg,
                                            act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(self.channels + c1_channels,
                                         self.channels,
                                         3,
                                         padding=1,
                                         norm_cfg=self.norm_cfg,
                                         act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(self.channels,
                                         self.channels,
                                         3,
                                         padding=1,
                                         norm_cfg=self.norm_cfg,
                                         act_cfg=self.act_cfg))

    def execute(self, inputs):
        """execute function."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(self.image_pool(x),
                   size=x.size()[2:],
                   mode='bilinear',
                   align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = jt.concat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(input=output,
                            size=c1_output.shape[2:],
                            mode='bilinear',
                            align_corners=self.align_corners)
            output = jt.concat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        output = self.cls_seg(output)
        return output
