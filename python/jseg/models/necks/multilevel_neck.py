from jittor import nn
from jseg.utils.weight_init import xavier_init
from jseg.ops import resize
from jseg.utils.registry import NECKS
import collections


@NECKS.register_module()
class MultiLevelNeck(nn.Module):
    """MultiLevelNeck.

    A neck structure connect vit backbone and decoder_heads.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        scales (List[float]): Scale factors for each input feature map.
            Default: [0.5, 1, 2, 4]
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self, in_channels, out_channels, scales=[0.5, 1, 2, 4]):
        super(MultiLevelNeck, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.num_outs = len(scales)
        self.lateral_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for in_channel in in_channels:

            self.lateral_convs.append(
                nn.Sequential(
                    collections.OrderedDict([
                        ('conv', nn.Conv(in_channel, out_channels, 1)),
                    ])))

        for _ in range(self.num_outs):
            self.convs.append(
                nn.Sequential(
                    collections.OrderedDict([('conv',
                                              nn.Conv(out_channels,
                                                      out_channels,
                                                      3,
                                                      padding=1))])))

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def execute(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inputs = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # for len(inputs) not equal to self.num_outs
        if len(inputs) == 1:
            inputs = [inputs[0] for _ in range(self.num_outs)]
        outs = []
        for i in range(self.num_outs):
            x_resize = resize(inputs[i],
                              scale_factor=self.scales[i],
                              mode='bilinear')
            outs.append(self.convs[i](x_resize))
        return tuple(outs)
