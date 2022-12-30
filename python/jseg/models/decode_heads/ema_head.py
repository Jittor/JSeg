import math

import jittor as jt
from jittor import nn
from jseg.bricks import ConvModule

from jseg.utils.registry import HEADS
from .decode_head import BaseDecodeHead


class EMAModule(nn.Module):

    def __init__(self, channels, num_bases, num_stages, momentum):
        super(EMAModule, self).__init__()
        assert num_stages >= 1, 'num_stages must be at least 1!'
        self.num_bases = num_bases
        self.num_stages = num_stages
        self.momentum = momentum

        bases = jt.normal(0,
                          math.sqrt(2. / self.num_bases),
                          size=(1, channels, self.num_bases))
        # [1, channels, num_bases]
        bases = jt.normalize(bases, dim=1, p=2)
        self.bases = bases

    def execute(self, feats):
        batch_size, channels, height, width = feats.size()
        # [batch_size, channels, height*width]
        feats = feats.view(batch_size, channels, height * width)
        # [batch_size, channels, num_bases]
        bases = self.bases.repeat(batch_size, 1, 1)

        with jt.no_grad():
            for i in range(self.num_stages):
                # [batch_size, height*width, num_bases]
                attention = jt.linalg.einsum('bcn,bck->bnk', feats, bases)
                attention = nn.softmax(attention, dim=2)
                # l1 norm
                attention_normed = jt.normalize(attention, dim=1, p=1)
                # [batch_size, channels, num_bases]
                bases = jt.linalg.einsum('bcn,bnk->bck', feats,
                                         attention_normed)
                # l2 norm
                bases = jt.normalize(bases, dim=1, p=2)

        feats_recon = jt.linalg.einsum('bck,bnk->bcn', bases, attention)
        feats_recon = feats_recon.view(batch_size, channels, height, width)

        if self.train():
            bases = bases.mean(dim=0, keepdim=True)
            # l2 norm
            bases = jt.normalize(bases, dim=1, p=2)
            self.bases = (1 -
                          self.momentum) * self.bases + self.momentum * bases

        return feats_recon


@HEADS.register_module()
class EMAHead(BaseDecodeHead):

    def __init__(self,
                 ema_channels,
                 num_bases,
                 num_stages,
                 concat_input=True,
                 momentum=0.1,
                 **kwargs):
        super(EMAHead, self).__init__(**kwargs)
        self.ema_channels = ema_channels
        self.num_bases = num_bases
        self.num_stages = num_stages
        self.concat_input = concat_input
        self.momentum = momentum
        self.ema_module = EMAModule(self.ema_channels, self.num_bases,
                                    self.num_stages, self.momentum)

        self.ema_in_conv = ConvModule(self.in_channels,
                                      self.ema_channels,
                                      3,
                                      padding=1,
                                      conv_cfg=self.conv_cfg,
                                      norm_cfg=self.norm_cfg,
                                      act_cfg=self.act_cfg)
        # project (0, inf) -> (-inf, inf)
        self.ema_mid_conv = ConvModule(self.ema_channels,
                                       self.ema_channels,
                                       1,
                                       conv_cfg=self.conv_cfg,
                                       norm_cfg=None,
                                       act_cfg=None)
        for param in self.ema_mid_conv.parameters():
            param.requires_grad = False

        self.ema_out_conv = ConvModule(self.ema_channels,
                                       self.ema_channels,
                                       1,
                                       conv_cfg=self.conv_cfg,
                                       norm_cfg=self.norm_cfg,
                                       act_cfg=None)
        self.bottleneck = ConvModule(self.ema_channels,
                                     self.channels,
                                     3,
                                     padding=1,
                                     conv_cfg=self.conv_cfg,
                                     norm_cfg=self.norm_cfg,
                                     act_cfg=self.act_cfg)
        if self.concat_input:
            self.conv_cat = ConvModule(self.in_channels + self.channels,
                                       self.channels,
                                       kernel_size=3,
                                       padding=1,
                                       conv_cfg=self.conv_cfg,
                                       norm_cfg=self.norm_cfg,
                                       act_cfg=self.act_cfg)

    def execute(self, inputs):
        x = self._transform_inputs(inputs)
        feats = self.ema_in_conv(x)
        identity = feats
        feats = self.ema_mid_conv(feats)
        recon = self.ema_module(feats)
        recon = nn.relu(recon)
        recon = self.ema_out_conv(recon)
        output = nn.relu(identity + recon)
        output = self.bottleneck(output)
        if self.concat_input:
            output = self.conv_cat(jt.concat([x, output], dim=1))
        output = self.cls_seg(output)
        return output
