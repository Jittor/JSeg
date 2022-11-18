from jittor import nn
from jseg.utils.weight_init import kaiming_init, constant_init
from typing import Tuple, Union


class ConvModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 padding=0,
                 bias=False):
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding,
                              dilation,
                              groups,
                              bias=bias)
        self.bn = nn.BatchNorm(out_channels)
        self.init_weights()

    def init_weights(self):
        kaiming_init(self.conv, a=0)
        constant_init(self.bn, 1, bias=0)

    def execute(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = nn.relu(x)
        return x


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1):
        super().__init__()

        # depthwise convolution
        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels)

        self.pointwise_conv = ConvModule(
            in_channels,
            out_channels,
            1)

    def execute(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
