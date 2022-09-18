import jittor as jt
from jittor import nn
from jseg.ops import ConvModule

from jseg.utils.registry import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class FCNHead(BaseDecodeHead):
    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(self.in_channels,
                       self.channels,
                       kernel_size=kernel_size,
                       padding=conv_padding,
                       dilation=dilation))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(self.channels,
                           self.channels,
                           kernel_size=kernel_size,
                           padding=conv_padding,
                           dilation=dilation))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(self.in_channels + self.channels,
                                       self.channels,
                                       kernel_size=kernel_size,
                                       padding=kernel_size // 2)

    def _execute_feature(self, inputs):
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(jt.concat([x, feats], dim=1))
        return feats

    def execute(self, inputs):
        output = self._execute_feature(inputs)
        output = self.cls_seg(output)
        return output
