import math
import jittor as jt
from jittor import nn

from ..losses import accuracy
from jseg.utils.registry import HEADS
from .decode_head import BaseDecodeHead
from jseg.utils.weight_init import constant_init, trunc_normal_init
from jseg.ops.cliprc_ops import TPN_Decoder, TPN_DecoderLayer, RecoveryDecoder


@HEADS.register_module()
class ATMSingleHeadSeg(BaseDecodeHead):

    def __init__(
        self,
        img_size,
        in_channels,
        seen_idx,
        all_idx,
        embed_dims=768,
        num_layers=3,
        num_heads=8,
        use_stages=1,
        use_proj=True,
        crop_train=False,
        recovery_decoder_num_layers=1,
        **kwargs,
    ):
        super(ATMSingleHeadSeg, self).__init__(in_channels=in_channels,
                                               **kwargs)

        self.image_size = img_size
        self.use_stages = use_stages
        self.crop_train = crop_train
        self.seen_idx = seen_idx
        self.all_idx = all_idx
        nhead = num_heads
        dim = embed_dims
        self.num_layers = num_layers
        input_proj = []
        proj_norm = []

        self.unseen_idx = self.all_idx.copy()
        for i_idx in self.seen_idx:
            self.unseen_idx.remove(i_idx)

        for i in range(self.use_stages):
            # FC layer to change ch
            if use_proj:
                proj = nn.Linear(self.in_channels, dim)
                trunc_normal_init(proj, std=.02)
            else:
                proj = nn.Identity()
            self.add_module("input_proj_{}".format(i + 1), proj)
            input_proj.append(proj)
            # norm layer
            if use_proj:
                norm = nn.LayerNorm(dim)
            else:
                norm = nn.Identity()
            self.add_module("proj_norm_{}".format(i + 1), norm)
            proj_norm.append(norm)
            # decoder layer
            decoder_layer = [
                TPN_DecoderLayer(d_model=dim,
                                 nhead=nhead,
                                 dim_feedforward=dim * 4)
                for i in range(num_layers)
            ]
            decoder = TPN_Decoder(decoder_layer, num_layers)

        self.input_proj = input_proj
        self.proj_norm = proj_norm
        self.decoder_q = decoder

        decoder_layer_v = [
            TPN_DecoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim * 4)
            for i in range(num_layers)
        ]

        self.decoder_v = TPN_Decoder(decoder_layer_v, num_layers)

        self.recovery_decoder = RecoveryDecoder(dim, nhead,
                                                recovery_decoder_num_layers)

        delattr(self, 'conv_seg')
        self.lateral_proj = nn.Linear(dim * 3, dim)
        self.q_proj = nn.Linear(dim * 2, dim)

    def add_module(self, name, module):
        setattr(self, name, module)
    
    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)
        # init self.decoder_v.layers[*].linear2.weight to zero
        for i in range(self.num_layers):
            constant_init(self.decoder_v.layers[i].linear2, val=0)

    def execute_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      self_training=False,
                      st_mask=None):
        seg_logits = self.execute(inputs)

        if self_training:
            pseudo_semantic_masks = seg_logits['pred_masks'].clone().detach(
            ).sigmoid()
            pseudo_semantic_masks[:, self.seen_idx, :, :] = -1
            pseudo_semantic_seg = pseudo_semantic_masks.argmax(
                dim=1).unsqueeze(1)
            # generate pseudo labels for "transductive" setting
            gt_semantic_seg[gt_semantic_seg == -1] = pseudo_semantic_seg[
                gt_semantic_seg == -1]
            gt_semantic_seg[gt_semantic_seg == -1] = 255
            losses = self.losses(seg_logits, gt_semantic_seg)
        else:
            gt_semantic_seg[gt_semantic_seg == -1] = 255
            losses = self.losses(seg_logits, gt_semantic_seg)

        return losses

    def execute_test(self, inputs, img_metas, test_cfg, self_training):
        return self.execute(inputs, self_training)

    def execute(self, inputs_both, self_training=None):
        x = []
        laterals = []
        attns = []
        maps_size = []
        qs = []
        out = {}

        inputs = inputs_both[0][0]
        cls_token = inputs_both[0][1]
        text_token = inputs_both[1]
        region_level_bridge = inputs_both[0][2]

        _, _, h_rlb, w_rlb = region_level_bridge.size()
        bs, d, H, W = inputs[0].size()

        for stage_ in inputs[:self.use_stages]:
            x.append(self.d4_to_d3(stage_) if stage_.dim() > 3 else stage_)
        x.reverse()

        for idx, (x_, proj_,
                  norm_) in enumerate(zip(x, self.input_proj, self.proj_norm)):
            lateral = norm_(proj_(x_))
            if idx == 0:
                laterals.append(lateral)
            else:
                if laterals[idx - 1].size()[1] == lateral.size()[1]:
                    laterals.append(lateral + laterals[idx - 1])
                else:
                    # nearest interpolate
                    l_ = self.d3_to_d4(laterals[idx - 1])
                    l_ = nn.interpolate(l_, scale_factor=2, mode="nearest")
                    l_ = self.d4_to_d3(l_)
                    laterals.append(l_ + lateral)

        lateral = laterals[-1]
        ori_lateral = lateral.clone()  # for recovery loss

        # Region Alignment Module
        q = self.combine_token(region_level_bridge, cls_token, text_token)

        q_ori = q.clone()  # for recovery loss

        q = self.q_proj(q)

        cls_token = cls_token.unsqueeze(1).expand(bs, lateral.size()[1], -1)
        region_level_bridge = nn.interpolate(region_level_bridge,
                                             size=(H, W),
                                             mode='bilinear',
                                             align_corners=False).reshape(
                                                 bs, d, H * W).transpose(1, 2)
        lateral = jt.concat((region_level_bridge, cls_token, lateral), dim=-1)

        lateral = self.lateral_proj(lateral)

        # semantic segmentation decoder
        qs = []
        laterals = []
        for idx in range(self.num_layers):
            q, lateral = self.decoder_q.layers[idx](
                q, lateral), self.decoder_v.layers[idx](lateral, q)
            qs.append(q)
            laterals.append(lateral)

        attns = []
        attn = qs[-1] @ laterals[-1].transpose(-2, -1)
        attn = attn.transpose(-1, -2)
        attn = self.d3_to_d4(attn)
        maps_size.append(attn.size()[-2:])
        attns.append(attn)

        # for recovery loss
        if self.training:
            out['ori_q'] = q_ori
            out['ori_lateral'] = ori_lateral
            q, lateral = self.recovery_decoder(q, lateral)
            out['q'] = q
            out['lateral'] = lateral

        outputs_seg_masks = []
        size = maps_size[-1]

        for i in range(len(attns)):
            attns[i] = self.fusion_attn_map(attns[i], text_token.shape[0],
                                            h_rlb * w_rlb)

        for i, attn in enumerate(attns):
            outputs_seg_masks.append(
                nn.interpolate(attn,
                               size=size,
                               mode='bilinear',
                               align_corners=False))

        pred = nn.interpolate(outputs_seg_masks[-1],
                              size=(self.image_size, self.image_size),
                              mode='bilinear',
                              align_corners=False)

        out["pred_masks"] = pred

        if self.training:
            outputs_seg_masks = jt.stack(outputs_seg_masks, dim=0)
        else:
            if self_training:
                out["pred"] = self.semantic_inference(out["pred_masks"],
                                                      self.seen_idx)
            else:
                out["pred"] = self.semantic_inference(out["pred_masks"],
                                                      self.seen_idx, 0.1)
            return out["pred"]
        return out

    def combine_token(self, region_level_bridge, cls_token, text_token):
        '''
        region_level_bridge: bs, d, h', w'
        cls_token: bs, d
        text_token: bs, c, d
        '''
        b, d, _, _ = region_level_bridge.size()
        region_level_bridge = region_level_bridge.reshape(b, d,
                                                          -1).transpose(1, 2)
        region_level_bridge_size = region_level_bridge.size()[1]
        text_token = text_token.unsqueeze(0).expand(b, -1, -1)
        cls_token_hw = cls_token.unsqueeze(1).expand(-1,
                                                     region_level_bridge_size,
                                                     -1)
        rlb_text_token = jt.linalg.einsum("bld,bcd->blcd",
                                          region_level_bridge + cls_token_hw,
                                          text_token)
        rlb_text_token = rlb_text_token.reshape(b, -1, d)
        text_token_hw = text_token.unsqueeze(1).expand(
            -1, region_level_bridge_size, -1, -1).reshape(b, -1, d)
        rlb_text_token = jt.concat((rlb_text_token, text_token_hw),
                                   dim=-1)  # b, l, c, d

        return rlb_text_token

    def fusion_attn_map(self, attn_map, class_num, region_level_bridge_size):
        '''
        attn_map: bs, class_num*region_level_bridge_size, h, w
        class_num: int
        region_level_bridge_size: int
        '''
        _, n_c, _, _ = attn_map.size()
        rlb_txt_attn = attn_map[:, :class_num, :, :]
        for i in range(1, region_level_bridge_size):
            rlb_txt_attn += attn_map[:,
                                     i * class_num:(i + 1) * class_num, :, :]
        return rlb_txt_attn / (n_c / class_num)

    def semantic_inference(self, mask_pred, seen_idx, weight=0.0):
        mask_pred = mask_pred.sigmoid()
        mask_pred[:, seen_idx] = mask_pred[:, seen_idx] - weight
        return mask_pred

    def d3_to_d4(self, t):
        n, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)

    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        if isinstance(seg_logit, dict):
            # atm loss
            seg_label = seg_label.squeeze(1)

            loss = self.loss_decode(seg_logit,
                                    seg_label,
                                    ignore_index=self.ignore_index)

            loss['acc_seg'] = accuracy(seg_logit["pred_masks"],
                                       seg_label,
                                       ignore_index=self.ignore_index)
            return loss
