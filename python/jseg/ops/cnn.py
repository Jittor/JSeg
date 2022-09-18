import jittor as jt
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


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + jt.rand(shape, dtype=x.dtype)
    output = x.divide(keep_prob) * random_tensor.floor()
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        return drop_path(x, self.drop_prob, self.is_training())
