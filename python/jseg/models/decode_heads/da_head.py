import jittor as jt
from jittor import nn
from jseg.ops import ConvModule

from jseg.utils.general import add_prefix
from jseg.utils.registry import HEADS
from jseg.ops import Scale, SelfAttentionBlock
from .decode_head import BaseDecodeHead


class PAM(SelfAttentionBlock):
    def __init__(self, in_channels, channels):
        super(PAM, self).__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=1,
            key_query_norm=False,
            value_out_num_convs=1,
            value_out_norm=False,
            matmul_norm=False,
            with_out=False)

        self.gamma = Scale(0)

    def execute(self, x):
        out = super(PAM, self).execute(x, x)

        out = self.gamma(out) + x
        return out


class CAM(nn.Module):
    """Channel Attention Module (CAM)"""

    def __init__(self):
        super(CAM, self).__init__()
        self.gamma = Scale(0)

    def execute(self, x):
        batch_size, channels, height, width = x.size()
        proj_query = x.view(batch_size, channels, -1)
        proj_key = x.view(batch_size, channels, -1).permute(0, 2, 1)
        energy = nn.bmm(proj_query, proj_key)
        energy_new = jt.max(
            energy, -1, keepdims=True)[0].expand_as(energy) - energy
        attention = nn.softmax(energy_new, dim=-1)
        proj_value = x.view(batch_size, channels, -1)

        out = nn.bmm(attention, proj_value)
        out = out.view(batch_size, channels, height, width)

        out = self.gamma(out) + x
        return out


@HEADS.register_module()
class DAHead(BaseDecodeHead):
    def __init__(self, pam_channels, **kwargs):
        super(DAHead, self).__init__(**kwargs)
        self.pam_channels = pam_channels
        self.pam_in_conv = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1)
        self.pam = PAM(self.channels, pam_channels)
        self.pam_out_conv = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1)
        self.pam_conv_seg = nn.Conv2d(
            self.channels, self.num_classes, kernel_size=1)

        self.cam_in_conv = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1)
        self.cam = CAM()
        self.cam_out_conv = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1)
        self.cam_conv_seg = nn.Conv2d(
            self.channels, self.num_classes, kernel_size=1)

    def pam_cls_seg(self, feat):
        """PAM feature classification."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.pam_conv_seg(feat)
        return output

    def cam_cls_seg(self, feat):
        """CAM feature classification."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.cam_conv_seg(feat)
        return output

    def execute(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        pam_feat = self.pam_in_conv(x)
        pam_feat = self.pam(pam_feat)
        pam_feat = self.pam_out_conv(pam_feat)
        pam_out = self.pam_cls_seg(pam_feat)

        cam_feat = self.cam_in_conv(x)
        cam_feat = self.cam(cam_feat)
        cam_feat = self.cam_out_conv(cam_feat)
        cam_out = self.cam_cls_seg(cam_feat)

        feat_sum = pam_feat + cam_feat
        pam_cam_out = self.cls_seg(feat_sum)

        return pam_cam_out, pam_out, cam_out

    def execute_test(self, inputs, img_metas, test_cfg):
        return self.execute(inputs)[0]

    def losses(self, seg_logit, seg_label):
        """Compute ``pam_cam``, ``pam``, ``cam`` loss."""
        pam_cam_seg_logit, pam_seg_logit, cam_seg_logit = seg_logit
        loss = dict()
        loss.update(
            add_prefix(
                super(DAHead, self).losses(pam_cam_seg_logit, seg_label),
                'pam_cam'))
        loss.update(
            add_prefix(
                super(DAHead, self).losses(pam_seg_logit, seg_label), 'pam'))
        loss.update(
            add_prefix(
                super(DAHead, self).losses(cam_seg_logit, seg_label), 'cam'))
        return loss
