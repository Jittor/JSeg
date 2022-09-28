from jittor import nn
from jseg.utils.weight_init import kaiming_init, constant_init


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
