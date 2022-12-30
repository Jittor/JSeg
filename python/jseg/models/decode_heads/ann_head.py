import jittor as jt
from jittor import nn
from jseg.bricks import ConvModule

from jseg.utils.registry import HEADS
from jseg.ops import SelfAttentionBlock as _SelfAttentionBlock
from .decode_head import BaseDecodeHead


class PPMConcat(nn.ModuleList):

    def __init__(self, pool_scales=(1, 3, 6, 8)):
        super(PPMConcat, self).__init__(
            [nn.AdaptiveAvgPool2d(pool_scale) for pool_scale in pool_scales])

    def execute(self, feats):
        """execute function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(feats)
            ppm_outs.append(ppm_out.view(*feats.shape[:2], -1))
        concat_outs = jt.concat(ppm_outs, dim=2)
        return concat_outs


class SelfAttentionBlock(_SelfAttentionBlock):

    def __init__(self, low_in_channels, high_in_channels, channels,
                 out_channels, share_key_query, query_scale, key_pool_scales,
                 conv_cfg, norm_cfg, act_cfg):
        key_psp = PPMConcat(key_pool_scales)
        if query_scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=query_scale)
        else:
            query_downsample = None
        super(SelfAttentionBlock,
              self).__init__(key_in_channels=low_in_channels,
                             query_in_channels=high_in_channels,
                             channels=channels,
                             out_channels=out_channels,
                             share_key_query=share_key_query,
                             query_downsample=query_downsample,
                             key_downsample=key_psp,
                             key_query_num_convs=1,
                             key_query_norm=True,
                             value_out_num_convs=1,
                             value_out_norm=False,
                             matmul_norm=True,
                             with_out=True,
                             conv_cfg=conv_cfg,
                             norm_cfg=norm_cfg,
                             act_cfg=act_cfg)


class AFNB(nn.Module):

    def __init__(self, low_in_channels, high_in_channels, channels,
                 out_channels, query_scales, key_pool_scales, conv_cfg,
                 norm_cfg, act_cfg):
        super(AFNB, self).__init__()
        self.stages = nn.ModuleList()
        for query_scale in query_scales:
            self.stages.append(
                SelfAttentionBlock(low_in_channels=low_in_channels,
                                   high_in_channels=high_in_channels,
                                   channels=channels,
                                   out_channels=out_channels,
                                   share_key_query=False,
                                   query_scale=query_scale,
                                   key_pool_scales=key_pool_scales,
                                   conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg,
                                   act_cfg=act_cfg))
        self.bottleneck = ConvModule(out_channels + high_in_channels,
                                     out_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=None)

    def execute(self, low_feats, high_feats):
        """execute function."""
        priors = [stage(high_feats, low_feats) for stage in self.stages]
        context = jt.stack(priors, dim=0).sum(dim=0)
        output = self.bottleneck(jt.concat([context, high_feats], 1))
        return output


class APNB(nn.Module):

    def __init__(self, in_channels, channels, out_channels, query_scales,
                 key_pool_scales, conv_cfg, norm_cfg, act_cfg):
        super(APNB, self).__init__()
        self.stages = nn.ModuleList()
        for query_scale in query_scales:
            self.stages.append(
                SelfAttentionBlock(low_in_channels=in_channels,
                                   high_in_channels=in_channels,
                                   channels=channels,
                                   out_channels=out_channels,
                                   share_key_query=True,
                                   query_scale=query_scale,
                                   key_pool_scales=key_pool_scales,
                                   conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg,
                                   act_cfg=act_cfg))
        self.bottleneck = ConvModule(2 * in_channels,
                                     out_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

    def execute(self, feats):
        """execute function."""
        priors = [stage(feats, feats) for stage in self.stages]
        context = jt.stack(priors, dim=0).sum(dim=0)
        output = self.bottleneck(jt.concat([context, feats], 1))
        return output


@HEADS.register_module()
class ANNHead(BaseDecodeHead):

    def __init__(self,
                 project_channels,
                 query_scales=(1, ),
                 key_pool_scales=(1, 3, 6, 8),
                 **kwargs):
        super(ANNHead, self).__init__(input_transform='multiple_select',
                                      **kwargs)
        assert len(self.in_channels) == 2
        low_in_channels, high_in_channels = self.in_channels
        self.project_channels = project_channels
        self.fusion = AFNB(low_in_channels=low_in_channels,
                           high_in_channels=high_in_channels,
                           out_channels=high_in_channels,
                           channels=project_channels,
                           query_scales=query_scales,
                           key_pool_scales=key_pool_scales,
                           conv_cfg=self.conv_cfg,
                           norm_cfg=self.norm_cfg,
                           act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(high_in_channels,
                                     self.channels,
                                     3,
                                     padding=1,
                                     conv_cfg=self.conv_cfg,
                                     norm_cfg=self.norm_cfg,
                                     act_cfg=self.act_cfg)
        self.context = APNB(in_channels=self.channels,
                            out_channels=self.channels,
                            channels=project_channels,
                            query_scales=query_scales,
                            key_pool_scales=key_pool_scales,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg)

    def execute(self, inputs):
        """execute function."""
        low_feats, high_feats = self._transform_inputs(inputs)
        output = self.fusion(low_feats, high_feats)
        output = self.dropout(output)
        output = self.bottleneck(output)
        output = self.context(output)
        output = self.cls_seg(output)

        return output
