from jittor import nn
from jseg.bricks import ConvModule

from jseg.utils.helpers import make_divisible
from jseg.utils.general import is_tuple_of


class SELayer(nn.Module):

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'),
                          dict(type='HSigmoid', bias=3.0, divisor=6.0))):
        super(SELayer, self).__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(in_channels=channels,
                                out_channels=make_divisible(
                                    channels // ratio, 8),
                                kernel_size=1,
                                stride=1,
                                conv_cfg=conv_cfg,
                                act_cfg=act_cfg[0])
        self.conv2 = ConvModule(in_channels=make_divisible(
            channels // ratio, 8),
                                out_channels=channels,
                                kernel_size=1,
                                stride=1,
                                conv_cfg=conv_cfg,
                                act_cfg=act_cfg[1])

    def execute(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out
