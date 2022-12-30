import math

import jittor as jt
from jittor import nn
from ..utils.embed import PatchEmbed, FFN
from jseg.utils.weight_init import kaiming_init, constant_init, trunc_normal_
from jseg.utils.helpers import to_2tuple
from jseg.bricks import build_norm_layer
from jittor.nn import BatchNorm as _BatchNorm

from jseg.ops import resize, MultiheadAttention
from jseg.utils.registry import BACKBONES


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedexecute_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 attn_cfg=dict(),
                 ffn_cfg=dict()):
        super(TransformerEncoderLayer, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(norm_cfg,
                                                  embed_dims,
                                                  postfix=1)
        setattr(self, self.norm1_name, norm1)

        attn_cfg.update(
            dict(embed_dims=embed_dims,
                 num_heads=num_heads,
                 attn_drop=attn_drop_rate,
                 proj_drop=drop_rate,
                 batch_first=batch_first,
                 bias=qkv_bias))

        self.build_attn(attn_cfg)

        self.norm2_name, norm2 = build_norm_layer(norm_cfg,
                                                  embed_dims,
                                                  postfix=2)
        setattr(self, self.norm2_name, norm2)

        ffn_cfg.update(
            dict(embed_dims=embed_dims,
                 feedexecute_channels=feedexecute_channels,
                 num_fcs=num_fcs,
                 ffn_drop=drop_rate,
                 dropout_layer=dict(type='DropPath', p=drop_path_rate)
                 if drop_path_rate > 0 else None,
                 act_cfg=act_cfg))
        self.build_ffn(ffn_cfg)

    def build_attn(self, attn_cfg):
        self.attn = MultiheadAttention(**attn_cfg)

    def build_ffn(self, ffn_cfg):
        self.ffn = FFN(**ffn_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def execute(self, x):

        def _inner_execute(x):
            x = self.attn(self.norm1(x), identity=x)
            x = self.ffn(self.norm2(x), identity=x)
            return x

        x = _inner_execute(x)
        return x


@BACKBONES.register_module()
class VisionTransformer(nn.Module):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4,
                 out_indices=-1,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 with_cls_token=True,
                 output_cls_token=False,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 patch_norm=False,
                 final_norm=False,
                 interpolate_mode='bicubic',
                 num_fcs=2,
                 norm_eval=False):
        super(VisionTransformer, self).__init__()

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'

        self.img_size = img_size
        self.patch_size = patch_size
        self.interpolate_mode = interpolate_mode
        self.norm_eval = norm_eval

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            padding='corner',
            norm_cfg=norm_cfg if patch_norm else None)

        num_patches = (img_size[0] // patch_size) * \
            (img_size[1] // patch_size)

        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = jt.zeros((1, 1, embed_dims))
        self.pos_embed = jt.zeros((1, num_patches + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = num_layers - 1
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')

        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, num_layers)
               ]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(embed_dims=embed_dims,
                                        num_heads=num_heads,
                                        feedexecute_channels=mlp_ratio *
                                        embed_dims,
                                        attn_drop_rate=attn_drop_rate,
                                        drop_rate=drop_rate,
                                        drop_path_rate=dpr[i],
                                        num_fcs=num_fcs,
                                        qkv_bias=qkv_bias,
                                        act_cfg=act_cfg,
                                        norm_cfg=norm_cfg,
                                        batch_first=True))

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(norm_cfg,
                                                      embed_dims,
                                                      postfix=1)
            setattr(self, self.norm1_name, norm1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            trunc_normal_(self.pos_embed, std=.02)
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
        elif isinstance(pretrained, str):
            checkpoint = jt.load(pretrained)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            if 'pos_embed' in state_dict.keys():
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    h, w = self.img_size
                    pos_size = int(
                        math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                    state_dict['pos_embed'] = self.resize_pos_embed(
                        state_dict['pos_embed'],
                        (h // self.patch_size, w // self.patch_size),
                        (pos_size, pos_size), self.interpolate_mode)
            self.load_parameters(state_dict)

    def _pos_embeding(self, patched_img, hw_shape, pos_embed):
        assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
            'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            if pos_len == (self.img_size[0] // self.patch_size) * (
                    self.img_size[1] // self.patch_size) + 1:
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(
                    'Unexpected shape of pos_embed, got {}.'.format(
                        pos_embed.shape))
            pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
                                              (pos_h, pos_w),
                                              self.interpolate_mode)
        return self.drop_after_pos(patched_img + pos_embed)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        if not isinstance(pos_embed, jt.Var):
            pos_embed = jt.Var(pos_embed)
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = resize(pos_embed_weight,
                                  size=input_shpae,
                                  align_corners=False,
                                  mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = jt.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = jt.concat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def execute(self, inputs):
        B = inputs.shape[0]
        x, hw_shape = self.patch_embed(inputs)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = jt.concat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                if self.with_cls_token:
                    # Remove class token and reshape token for decoder head
                    out = x[:, 1:]
                else:
                    out = x
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1],
                                  C).permute(0, 3, 1, 2)
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        super(VisionTransformer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()
