import jittor as jt
from jittor import nn
from jseg.utils.registry import BACKBONES


class ResLayer(nn.Sequential):
    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 dilation=1,
                 avg_down=False,
                 multi_grid=None,
                 contract_dilation=False,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(kernel_size=stride,
                                 stride=stride,
                                 ceil_mode=True,
                                 count_include_pad=False))
            downsample.extend([
                nn.Conv2d(inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=conv_stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if multi_grid is None:
            if dilation > 1 and contract_dilation:
                first_dilation = dilation // 2
            else:
                first_dilation = dilation
        else:
            first_dilation = multi_grid[0]
        layers.append(
            block(inplanes=inplanes,
                  planes=planes,
                  stride=stride,
                  dilation=first_dilation,
                  downsample=downsample,
                  **kwargs))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    dilation=dilation if multi_grid is None else multi_grid[i],
                    **kwargs))
        super(ResLayer, self).__init__(*layers)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 dcn=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv1 = nn.Conv2d(inplanes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation,
                               dilation=dilation,
                               bias=False)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               padding=1,
                               bias=False)

        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def execute(self, x):
        def _inner_execute(x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        out = _inner_execute(x)

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 dcn=None):
        super(Bottleneck, self).__init__()
        assert dcn is None or isinstance(dcn, dict)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.dcn = dcn
        self.with_dcn = dcn is not None

        self.conv1_stride = 1
        self.conv2_stride = stride

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.conv1 = nn.Conv2d(inplanes,
                               planes,
                               kernel_size=1,
                               stride=self.conv1_stride,
                               bias=False)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=self.conv2_stride,
                                   padding=dilation,
                                   dilation=dilation,
                                   bias=False)
        else:
            self.conv2 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=self.conv2_stride,
                                   padding=dilation,
                                   dilation=dilation,
                                   bias=False)

        self.conv3 = nn.Conv2d(planes,
                               planes * self.expansion,
                               kernel_size=1,
                               bias=False)

        self.relu = nn.ReLU()
        self.downsample = downsample

    def execute(self, x):
        def _inner_execute(x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

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
class ResNet(nn.Module):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 norm_eval=False,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 multi_grid=None,
                 contract_dilation=False,
                 zero_init_residual=True):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.multi_grid = multi_grid
        self.contract_dilation = contract_dilation
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            # multi grid is applied to last layer only
            stage_multi_grid = multi_grid if i == len(
                self.stage_blocks) - 1 else None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                avg_down=self.avg_down,
                dcn=dcn,
                multi_grid=stage_multi_grid,
                contract_dilation=contract_dilation)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i+1}'
            setattr(self, layer_name, res_layer)
            # self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    def _make_stem_layer(self, in_channels, stem_channels):
        """Make stem layer for ResNet."""
        if self.deep_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels,
                          stem_channels // 2,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias=False), nn.BatchNorm2d(stem_channels // 2),
                nn.ReLU(),
                nn.Conv2d(stem_channels // 2,
                          stem_channels // 2,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False), nn.BatchNorm2d(stem_channels // 2),
                nn.ReLU(),
                nn.Conv2d(stem_channels // 2,
                          stem_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False), nn.BatchNorm2d(stem_channels),
                nn.ReLU())
        else:
            self.conv1 = nn.Conv2d(in_channels,
                                   stem_channels,
                                   kernel_size=7,
                                   stride=2,
                                   padding=3,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(stem_channels),
            self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.bn1.eval()
                for m in [self.conv1, self.bn1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            self.load_parameters(jt.load(pretrained))

    def execute(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


@BACKBONES.register_module()
class ResNetV1c(ResNet):
    def __init__(self, **kwargs):
        super(ResNetV1c, self).__init__(deep_stem=True,
                                        avg_down=False,
                                        **kwargs)


@BACKBONES.register_module()
class ResNetV1d(ResNet):
    def __init__(self, **kwargs):
        super(ResNetV1d, self).__init__(deep_stem=True,
                                        avg_down=True,
                                        **kwargs)
