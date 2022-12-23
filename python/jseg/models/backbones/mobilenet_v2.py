import jittor as jt
from jittor import nn
from jseg.bricks import ConvModule
from jittor.nn import BatchNorm as _BatchNorm

from jseg.utils.registry import BACKBONES
from ..utils.inverted_residual import InvertedResidual
from jseg.utils.helpers import make_divisible


@BACKBONES.register_module()
class MobileNetV2(nn.Module):
    # Parameters to build layers. 3 parameters are needed to construct a
    # layer, from left to right: expand_ratio, channel, num_blocks.
    arch_settings = [[1, 16, 1], [6, 24, 2], [6, 32, 3], [6, 64, 4],
                     [6, 96, 3], [6, 160, 3], [6, 320, 1]]

    def __init__(self,
                 widen_factor=1.,
                 strides=(1, 2, 2, 2, 1, 2, 1),
                 dilations=(1, 1, 1, 1, 1, 1, 1),
                 out_indices=(1, 2, 4, 6),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 norm_eval=False):
        super(MobileNetV2, self).__init__()

        self.widen_factor = widen_factor
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == len(self.arch_settings)
        self.out_indices = out_indices
        for index in out_indices:
            if index not in range(0, 7):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 7). But received {index}')

        if frozen_stages not in range(-1, 7):
            raise ValueError('frozen_stages must be in range(-1, 7). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval

        self.in_channels = make_divisible(32 * widen_factor, 8)

        self.conv1 = ConvModule(in_channels=3,
                                out_channels=self.in_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                conv_cfg=self.conv_cfg,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg)

        self.layers = []

        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks = layer_cfg
            stride = self.strides[i]
            dilation = self.dilations[i]
            out_channels = make_divisible(channel * widen_factor, 8)
            inverted_res_layer = self.make_layer(out_channels=out_channels,
                                                 num_blocks=num_blocks,
                                                 stride=stride,
                                                 dilation=dilation,
                                                 expand_ratio=expand_ratio)
            layer_name = f'layer{i + 1}'
            setattr(self, layer_name, inverted_res_layer)
            self.layers.append(layer_name)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            self.load_parameters(jt.load(pretrained))

    def make_layer(self, out_channels, num_blocks, stride, dilation,
                   expand_ratio):
        layers = []
        for i in range(num_blocks):
            layers.append(
                InvertedResidual(self.in_channels,
                                 out_channels,
                                 stride if i == 0 else 1,
                                 expand_ratio=expand_ratio,
                                 dilation=dilation if i == 0 else 1,
                                 conv_cfg=self.conv_cfg,
                                 norm_cfg=self.norm_cfg,
                                 act_cfg=self.act_cfg))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def execute(self, x):
        x = self.conv1(x)

        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(MobileNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
