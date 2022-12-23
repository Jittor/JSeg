from jseg.bricks import ConvModule
from jittor import nn

from .se_layer import SELayer


class InvertedResidual(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 **kwargs):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2], f'stride must in [1, 2]. ' \
            f'But received {stride}.'
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **kwargs))
        layers.extend([
            ConvModule(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=hidden_dim,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                **kwargs),
            ConvModule(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
                **kwargs)
        ])
        self.conv = nn.Sequential(*layers)

    def execute(self, x):

        def _inner_execute(x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)

        out = _inner_execute(x)

        return out


class InvertedResidualV3(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 kernel_size=3,
                 stride=1,
                 se_cfg=None,
                 with_expand_conv=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(InvertedResidualV3, self).__init__()
        self.with_res_shortcut = (stride == 1 and in_channels == out_channels)
        assert stride in [1, 2]
        self.with_se = se_cfg is not None
        self.with_expand_conv = with_expand_conv

        if self.with_se:
            assert isinstance(se_cfg, dict)
        if not self.with_expand_conv:
            assert mid_channels == in_channels

        if self.with_expand_conv:
            self.expand_conv = ConvModule(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.depthwise_conv = ConvModule(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=mid_channels,
            conv_cfg=dict(
                type='Conv2dAdaptivePadding') if stride == 2 else conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        if self.with_se:
            self.se = SELayer(**se_cfg)

        self.linear_conv = ConvModule(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def execute(self, x):

        def _inner_execute(x):
            out = x

            if self.with_expand_conv:
                out = self.expand_conv(out)

            out = self.depthwise_conv(out)

            if self.with_se:
                out = self.se(out)

            out = self.linear_conv(out)

            if self.with_res_shortcut:
                return x + out
            else:
                return out

        out = _inner_execute(x)

        return out
