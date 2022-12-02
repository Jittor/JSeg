import numpy as np
import jittor as jt
from jittor import nn
from jseg.bricks import build_norm_layer, build_dropout
from jseg.utils.weight_init import (constant_init, kaiming_init, trunc_normal_)
from jseg.utils.helpers import to_2tuple
from jittor.nn import BatchNorm as _BatchNorm

from jseg.utils.registry import BACKBONES
from ..utils.embed import PatchEmbed
from .vit import TransformerEncoderLayer as VisionTransformerEncoderLayer

try:
    from scipy import interpolate
except ImportError:
    interpolate = None


class BEiTAttention(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 bias='qv_bias',
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.bias = bias
        self.scale = qk_scale or head_embed_dims**-0.5

        qkv_bias = bias
        if bias == 'qv_bias':
            self._init_qv_bias()
            qkv_bias = False

        self.window_size = window_size
        self._init_rel_pos_embedding()

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

    def _init_qv_bias(self):
        self.q_bias = jt.zeros((self.embed_dims))
        self.v_bias = jt.zeros((self.embed_dims))

    def _init_rel_pos_embedding(self):
        Wh, Ww = self.window_size
        # cls to token & token 2 cls & cls to cls
        self.num_relative_distance = (2 * Wh - 1) * (2 * Ww - 1) + 3
        # relative_position_bias_table shape is (2*Wh-1 * 2*Ww-1 + 3, nH)
        self.relative_position_bias_table = jt.zeros(
            (self.num_relative_distance, self.num_heads))

        # get pair-wise relative position index for
        # each token inside the window
        coords_h = jt.arange(Wh)
        coords_w = jt.arange(Ww)
        # coords shape is (2, Wh, Ww)
        coords = jt.stack(jt.meshgrid([coords_h, coords_w]))
        # coords_flatten shape is (2, Wh*Ww)
        coords_flatten = jt.flatten(coords, 1)
        relative_coords = (coords_flatten[:, :, None] -
                           coords_flatten[:, None, :])
        # relative_coords shape is (Wh*Ww, Wh*Ww, 2)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += Wh - 1
        relative_coords[:, :, 1] += Ww - 1
        relative_coords[:, :, 0] *= 2 * Ww - 1
        relative_position_index = jt.zeros(
            ((Wh * Ww + 1, ) * 2)).astype(relative_coords.dtype)
        # relative_position_index shape is (Wh*Ww, Wh*Ww)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.relative_position_index = relative_position_index

    def execute(self, x):
        """
        Args:
            x (Var): input features with shape of (num_windows*B, N, C).
        """
        B, N, C = x.shape
        if self.bias == 'qv_bias':
            k_bias = jt.zeros_like(self.v_bias)
            qkv_bias = jt.concat((self.q_bias, k_bias, self.v_bias))
            # linear
            qkv = x.matmul(self.qkv.weight.t()) + qkv_bias
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if self.relative_position_bias_table is not None:
            Wh = self.window_size[0]
            Ww = self.window_size[1]
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)].view(
                    Wh * Ww + 1, Wh * Ww + 1, -1)
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BEiTTransformerEncoderLayer(VisionTransformerEncoderLayer):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedexecute_channels,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 bias='qv_bias',
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 window_size=None,
                 attn_cfg=dict(),
                 ffn_cfg=dict(add_identity=False),
                 init_values=None):
        attn_cfg.update(dict(window_size=window_size, qk_scale=None))

        super(BEiTTransformerEncoderLayer,
              self).__init__(embed_dims=embed_dims,
                             num_heads=num_heads,
                             feedexecute_channels=feedexecute_channels,
                             attn_drop_rate=attn_drop_rate,
                             drop_path_rate=0.,
                             drop_rate=0.,
                             num_fcs=num_fcs,
                             qkv_bias=bias,
                             act_cfg=act_cfg,
                             norm_cfg=norm_cfg,
                             attn_cfg=attn_cfg,
                             ffn_cfg=ffn_cfg)

        # NOTE: drop path for stochastic depth, we shall see if
        # this is better than dropout here
        dropout_layer = dict(type='DropPath', p=drop_path_rate)
        self.drop_path = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        self.gamma_1 = init_values * jt.ones((embed_dims))
        self.gamma_2 = init_values * jt.ones((embed_dims))

    def build_attn(self, attn_cfg):
        self.attn = BEiTAttention(**attn_cfg)

    def execute(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        return x


@BACKBONES.register_module()
class BEiT(nn.Module):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4,
                 out_indices=-1,
                 qv_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 patch_norm=False,
                 final_norm=False,
                 num_fcs=2,
                 norm_eval=False,
                 pretrained=None,
                 init_values=0.1):
        super(BEiT, self).__init__()
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.norm_eval = norm_eval
        self.pretrained = pretrained
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.num_fcs = num_fcs
        self.qv_bias = qv_bias
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.patch_norm = patch_norm
        self.init_values = init_values
        self.window_size = (img_size[0] // patch_size,
                            img_size[1] // patch_size)
        self.patch_shape = self.window_size
        self.cls_token = jt.zeros((1, 1, embed_dims))

        self._build_patch_embedding()
        self._build_layers()

        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = num_layers - 1
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(norm_cfg,
                                                      embed_dims,
                                                      postfix=1)
            setattr(self, self.norm1_name, norm1)

    def _build_patch_embedding(self):
        """Build patch embedding layer."""
        self.patch_embed = PatchEmbed(
            in_channels=self.in_channels,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            norm_cfg=self.norm_cfg if self.patch_norm else None)

    def _build_layers(self):
        """Build transformer encoding layers."""

        dpr = [
            x.item()
            for x in jt.linspace(0, self.drop_path_rate, self.num_layers)
        ]
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                BEiTTransformerEncoderLayer(
                    embed_dims=self.embed_dims,
                    num_heads=self.num_heads,
                    feedexecute_channels=self.mlp_ratio * self.embed_dims,
                    attn_drop_rate=self.attn_drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=self.num_fcs,
                    bias='qv_bias' if self.qv_bias else False,
                    act_cfg=self.act_cfg,
                    norm_cfg=self.norm_cfg,
                    window_size=self.window_size,
                    init_values=self.init_values))

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _geometric_sequence_interpolation(self, src_size, dst_size, sequence,
                                          num):
        """Get new sequence via geometric sequence interpolation.

        Args:
            src_size (int): Pos_embedding size in pre-trained model.
            dst_size (int): Pos_embedding size in the current model.
            sequence (Var): The relative position bias of the pretrain
                model after removing the extra tokens.
            num (int): Number of attention heads.
        Returns:
            new_sequence (Var): Geometric sequence interpolate the
                pre-trained relative position bias to the size of
                the current model.
        """

        def geometric_progression(a, r, n):
            return a * (1.0 - r**n) / (1.0 - r)

        # Here is a binary function.
        left, right = 1.01, 1.5
        while right - left > 1e-6:
            q = (left + right) / 2.0
            gp = geometric_progression(1, q, src_size // 2)
            if gp > dst_size // 2:
                right = q
            else:
                left = q
        # The position of each interpolated point is determined
        # by the ratio obtained by dichotomy.
        dis = []
        cur = 1
        for i in range(src_size // 2):
            dis.append(cur)
            cur += q**(i + 1)
        r_ids = [-_ for _ in reversed(dis)]
        x = r_ids + [0] + dis
        y = r_ids + [0] + dis
        t = dst_size // 2.0
        dx = np.arange(-t, t + 0.1, 1.0)
        dy = np.arange(-t, t + 0.1, 1.0)
        # Interpolation functions are being executed and called.
        new_sequence = []
        for i in range(num):
            if isinstance(sequence, jt.Var):
                z = sequence[:, i].view(src_size, src_size).float().numpy()
                f = interpolate.interp2d(x, y, z, kind='cubic')
                new_sequence.append(jt.Var(f(dx, dy)).view(-1, 1))
            else:
                z = jt.Var(sequence[:, i]).view(src_size,
                                                src_size).float().numpy()
                f = interpolate.interp2d(x, y, z, kind='cubic')
                new_sequence.append(jt.Var(f(dx, dy)).view(-1, 1))

        new_sequence = jt.concat(new_sequence, dim=-1)
        return new_sequence

    def resize_rel_pos_embed(self, checkpoint):
        """Resize relative pos_embed weights.

        This function is modified from
        https://github.com/microsoft/unilm/blob/master/beit/semantic_segmentation/mmcv_custom/checkpoint.py.  # noqa: E501
        Copyright (c) Microsoft Corporation
        Licensed under the MIT License
        Args:
            checkpoint (dict): Key and value of the pretrain model.
        Returns:
            state_dict (dict): Interpolate the relative pos_embed weights
                in the pre-train model to the current model size.
        """
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        all_keys = list(state_dict.keys())
        for key in all_keys:
            if 'relative_position_index' in key:
                state_dict.pop(key)
            # In order to keep the center of pos_bias as consistent as
            # possible after interpolation, and vice versa in the edge
            # area, the geometric sequence interpolation method is adopted.
            if 'relative_position_bias_table' in key:
                rel_pos_bias = state_dict[key]
                if isinstance(rel_pos_bias, jt.Var):
                    src_num_pos, num_attn_heads = rel_pos_bias.size()
                    dst_num_pos, _ = self.state_dict()[key].size()
                else:
                    src_num_pos, num_attn_heads = rel_pos_bias.shape
                    dst_num_pos, _ = self.state_dict()[key].shape
                dst_patch_shape = self.patch_shape
                if dst_patch_shape[0] != dst_patch_shape[1]:
                    raise NotImplementedError()
                # Count the number of extra tokens.
                num_extra_tokens = dst_num_pos - (
                    dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
                src_size = int((src_num_pos - num_extra_tokens)**0.5)
                dst_size = int((dst_num_pos - num_extra_tokens)**0.5)
                if src_size != dst_size:
                    extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                    rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]
                    new_rel_pos_bias = self._geometric_sequence_interpolation(
                        src_size, dst_size, rel_pos_bias, num_attn_heads)
                    new_rel_pos_bias = jt.concat(
                        (new_rel_pos_bias, extra_tokens), dim=0)
                    state_dict[key] = new_rel_pos_bias

        return state_dict

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
            self.load_parameters(state_dict)

    def execute(self, inputs):
        B = inputs.shape[0]
        x, hw_shape = self.patch_embed(inputs)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = jt.concat((cls_tokens, x), dim=1)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                # Remove class token and reshape token for decoder head
                out = x[:, 1:]
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1],
                                  C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        super(BEiT, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()
