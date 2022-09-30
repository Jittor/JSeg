import math
import jittor as jt
from jittor import nn

from jseg.utils.registry import BACKBONES
from .resnet import ResLayer
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNetV1d


class RSoftmax(nn.Module):

    def __init__(self, radix, groups):
        super().__init__()
        self.radix = radix
        self.groups = groups

    def execute(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.groups, self.radix, -1).transpose(1, 2)
            x = nn.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = jt.sigmoid(x)
        return x


class SplitAttentionConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 radix=2,
                 reduction_factor=4,
                 dcn=None):
        super(SplitAttentionConv2d, self).__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.groups = groups
        self.channels = channels
        self.with_dcn = dcn is not None
        self.dcn = dcn
        self.conv = nn.Conv2d(in_channels,
                              channels * radix,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups * radix,
                              bias=False)
        self.bn0 = nn.BatchNorm2d(channels * radix)
        self.relu = nn.ReLU()
        self.fc1 = nn.Conv2d(
            channels,
            inter_channels,
            1,
            groups=self.groups,
        )
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.fc2 = nn.Conv2d(
            inter_channels,
            channels * radix,
            1,
            groups=self.groups,
        )

        self.rsoftmax = RSoftmax(radix, groups)
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d(1)

    def execute(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        batch = x.size(0)
        if self.radix > 1:
            splits = x.view(batch, self.radix, -1, *x.shape[2:])
            gap = splits.sum(dim=1)
        else:
            gap = x
        gap = self.adaptive_avg_pool2d(gap)
        gap = self.fc1(gap)

        gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = atten.view(batch, self.radix, -1, *atten.shape[2:])
            out = jt.sum(attens * splits, dim=1)
        else:
            out = atten * x
        return out


class Bottleneck(_Bottleneck):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 groups=1,
                 base_width=4,
                 base_channels=64,
                 radix=2,
                 reduction_factor=4,
                 avg_down_stride=True,
                 **kwargs):
        """Bottleneck block for ResNeSt."""
        super(Bottleneck, self).__init__(inplanes, planes, **kwargs)

        if groups == 1:
            width = self.planes
        else:
            width = math.floor(self.planes *
                               (base_width / base_channels)) * groups

        self.avg_down_stride = avg_down_stride and self.conv2_stride > 1

        self.bn1 = nn.BatchNorm2d(width)
        self.bn3 = nn.BatchNorm2d(self.planes * self.expansion)

        self.conv1 = nn.Conv2d(self.inplanes,
                               width,
                               kernel_size=1,
                               stride=self.conv1_stride,
                               bias=False)
        self.with_modulated_dcn = False
        self.conv2 = SplitAttentionConv2d(
            width,
            width,
            kernel_size=3,
            stride=1 if self.avg_down_stride else self.conv2_stride,
            padding=self.dilation,
            dilation=self.dilation,
            groups=groups,
            radix=radix,
            reduction_factor=reduction_factor,
            dcn=self.dcn)

        if self.avg_down_stride:
            self.avd_layer = nn.AvgPool2d(3, self.conv2_stride, padding=1)

        self.conv3 = nn.Conv2d(width,
                               self.planes * self.expansion,
                               kernel_size=1,
                               bias=False)

    def execute(self, x):

        def _inner_execute(x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)

            if self.avg_down_stride:
                out = self.avd_layer(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        out = _inner_execute(x)
        out = self.relu(out)

        return out


@BACKBONES.register_module()
class ResNeSt(ResNetV1d):
    """ResNeSt backbone.

    This backbone is the implementation of `ResNeSt:
    Split-Attention Networks <https://arxiv.org/abs/2004.08955>`_.

    Args:
        groups (int): Number of groups of Bottleneck. Default: 1
        base_width (int): Base width of Bottleneck. Default: 4
        radix (int): Radix of SpltAtConv2d. Default: 2
        reduction_factor (int): Reduction factor of inter_channels in
            SplitAttentionConv2d. Default: 4.
        avg_down_stride (bool): Whether to use average pool for stride in
            Bottleneck. Default: True.
        kwargs (dict): Keyword arguments for ResNet.
    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
        200: (Bottleneck, (3, 24, 36, 3))
    }

    def __init__(self,
                 groups=1,
                 base_width=4,
                 radix=2,
                 reduction_factor=4,
                 avg_down_stride=True,
                 **kwargs):
        self.groups = groups
        self.base_width = base_width
        self.radix = radix
        self.reduction_factor = reduction_factor
        self.avg_down_stride = avg_down_stride
        super(ResNeSt, self).__init__(**kwargs)

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(groups=self.groups,
                        base_width=self.base_width,
                        base_channels=self.base_channels,
                        radix=self.radix,
                        reduction_factor=self.reduction_factor,
                        avg_down_stride=self.avg_down_stride,
                        **kwargs)
