import math

import jittor as jt
from jittor import nn
from jseg.utils.weight_init import (constant_init, kaiming_init, trunc_normal_)
from jittor.nn import BatchNorm as _BatchNorm

from jseg.utils.registry import BACKBONES
from .beit import BEiT, BEiTAttention, BEiTTransformerEncoderLayer


class MAEAttention(BEiTAttention):
    """Multi-head self-attention with relative position bias used in MAE.

    This module is different from ``BEiTAttention`` by initializing the
    relative bias table with zeros.
    """

    def init_weights(self):
        """Initialize relative position bias with zeros."""

        # As MAE initializes relative position bias as zeros and this class
        # inherited from BEiT which initializes relative position bias
        # with `trunc_normal`, `init_weights` here does
        # nothing and just passes directly

        pass


class MAETransformerEncoderLayer(BEiTTransformerEncoderLayer):
    """Implements one encoder layer in Vision Transformer.

    This module is different from ``BEiTTransformerEncoderLayer`` by replacing
    ``BEiTAttention`` with ``MAEAttention``.
    """

    def build_attn(self, attn_cfg):
        self.attn = MAEAttention(**attn_cfg)


@BACKBONES.register_module()
class MAE(BEiT):
    """VisionTransformer with support for patch.

    Args:
        img_size (int | tuple): Input image size. Default: 224.
        patch_size (int): The patch size. Default: 16.
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): embedding dimension. Default: 768.
        num_layers (int): depth of transformer. Default: 12.
        num_heads (int): number of attention heads. Default: 12.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        out_indices (list | tuple | int): Output from which stages.
            Default: -1.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        patch_norm (bool): Whether to add a norm in PatchEmbed Block.
            Default: False.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Default: False.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        init_values (float): Initialize the values of Attention and FFN
            with learnable scaling. Defaults to 0.1.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4,
                 out_indices=-1,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 patch_norm=False,
                 final_norm=False,
                 num_fcs=2,
                 norm_eval=False,
                 init_values=0.1):
        super(MAE, self).__init__(img_size=img_size,
                                  patch_size=patch_size,
                                  in_channels=in_channels,
                                  embed_dims=embed_dims,
                                  num_layers=num_layers,
                                  num_heads=num_heads,
                                  mlp_ratio=mlp_ratio,
                                  out_indices=out_indices,
                                  qv_bias=False,
                                  attn_drop_rate=attn_drop_rate,
                                  drop_path_rate=drop_path_rate,
                                  norm_cfg=norm_cfg,
                                  act_cfg=act_cfg,
                                  patch_norm=patch_norm,
                                  final_norm=final_norm,
                                  num_fcs=num_fcs,
                                  norm_eval=norm_eval,
                                  init_values=init_values)

        self.cls_token = jt.zeros((1, 1, embed_dims))

        self.num_patches = self.patch_shape[0] * self.patch_shape[1]
        self.pos_embed = jt.zeros((1, self.num_patches + 1, embed_dims))

    def _build_layers(self):
        dpr = [
            x.item()
            for x in jt.linspace(0, self.drop_path_rate, self.num_layers)
        ]
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                MAETransformerEncoderLayer(
                    embed_dims=self.embed_dims,
                    num_heads=self.num_heads,
                    feedexecute_channels=self.mlp_ratio * self.embed_dims,
                    attn_drop_rate=self.attn_drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=self.num_fcs,
                    bias=True,
                    act_cfg=self.act_cfg,
                    norm_cfg=self.norm_cfg,
                    window_size=self.patch_shape,
                    init_values=self.init_values))

    def fix_init_weight(self):
        """Rescale the initialization according to layer id.

        This function is copied from  https://github.com/microsoft/unilm/blob/master/beit/modeling_pretrain.py. # noqa: E501
        Copyright (c) Microsoft Corporation
        Licensed under the MIT License
        """

        def rescale(param, layer_id):
            param /= math.sqrt(2.0 * layer_id)

        for layer_id, layer in enumerate(self.layers):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.ffn.layers[1].weight.data, layer_id + 1)

    def init_weights(self, pretrained=None):

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)
        self.fix_init_weight()

        if pretrained is None:
            trunc_normal_(self.cls_token, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            nn.init.normal_(m.bias, mean=0., std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)
        else:
            checkpoint = jt.load(pretrained)
            state_dict = self.resize_rel_pos_embed(checkpoint)
            state_dict = self.resize_abs_pos_embed(state_dict)
            self.load_parameters(state_dict)

    def resize_abs_pos_embed(self, state_dict):
        if 'pos_embed' in state_dict:
            pos_embed_checkpoint = state_dict['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_extra_tokens = self.pos_embed.shape[-2] - self.num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int(
                (pos_embed_checkpoint.shape[-2] - num_extra_tokens)**0.5)
            # height (== width) for the new position embedding
            new_size = int(self.num_patches**0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                if not isinstance(pos_tokens, jt.Var):
                    pos_tokens = jt.Var(pos_tokens)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
                                                embedding_size).permute(
                                                    0, 3, 1, 2)
                pos_tokens = nn.interpolate(
                    pos_tokens,
                    size=(new_size, new_size),
                    mode='bicubic',
                    align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = jt.concat((extra_tokens, pos_tokens), dim=1)
                state_dict['pos_embed'] = new_pos_embed
        return state_dict

    def execute(self, inputs):
        B = inputs.shape[0]

        x, hw_shape = self.patch_embed(inputs)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = jt.concat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                out = x[:, 1:]
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1],
                                  C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)
