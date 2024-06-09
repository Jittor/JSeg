import jittor as jt
from jittor import nn
from collections import OrderedDict
from .mha import MultiheadAttention
from jittor import Var
from typing import Optional
from jseg.utils.weight_init import constant_init, trunc_normal_init


class LayerNorm(nn.LayerNorm):

    def execute(self, x):
        ret = super().execute(x)
        return jt.type_as(ret, x)


class QuickGELU(nn.Module):

    def execute(self, x):
        return x * jt.sigmoid(1.702 * x)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        return nn.droppath(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class ResidualAttentionBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask=None,
                 drop_path=0.):
        super().__init__()

        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)),
                         ("gelu", QuickGELU()),
                         ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False,
                         attn_mask=self.attn_mask)[0]

    def attention_weight(self, x):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True,
                         attn_mask=self.attn_mask)[1]

    def execute(self, x, return_attention: bool = False):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask=None,
                 drop_path_rate=0.):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, layers)
               ]  # stochastic depth decay rule
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask, dpr[i])
            for i in range(layers)
        ])

    def execute(self, x):
        return self.resblocks(x)

    # ADDED
    def execute_attention(self, x):
        for index, layer in enumerate(self.resblocks):
            if index == len(self.resblocks) - 1:
                return layer(x, return_attention=True)
            x = layer(x)


# for decoder


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def execute(self, xq, xk, xv, return_memory=False):
        B, Nq, C = xq.size()  # 1, 21, 512
        Nk = xk.size()[1]
        Nv = xv.size()[1]

        q = self.q(xq).reshape(B, Nq, self.num_heads,
                               C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(xk).reshape(B, Nk, self.num_heads,
                               C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(xv).reshape(B, Nv, self.num_heads,
                               C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_save = attn.clone()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_memory:
            return x, attn_save.sum(dim=1) / self.num_heads, k.permute(
                0, 2, 1, 3).reshape(B, Nk, C), v.permute(0, 2, 1,
                                                         3).reshape(B, Nv, C)
        return x, attn_save.sum(dim=1) / self.num_heads


def _get_activation_fn(activation):
    if activation == "relu":
        return nn.relu
    elif activation == "gelu":
        return nn.gelu

    raise RuntimeError(
        "activation should be relu/gelu, not {}".format(activation))


class TransformerDecoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [decoder_layer[i] for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def execute(self,
                tgt: Var,
                memory: Var,
                tgt_mask: Optional[Var] = None,
                memory_mask: Optional[Var] = None,
                tgt_key_padding_mask: Optional[Var] = None,
                memory_key_padding_mask: Optional[Var] = None) -> Var:
        output = tgt

        for mod in self.layers:
            output = mod(output,
                         memory,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=nn.relu,
                 layer_norm_eps=1e-5,
                 batch_first=False,
                 norm_first=False) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model,
                                            nhead,
                                            dropout=dropout,
                                            batch_first=batch_first)
        self.multihead_attn = MultiheadAttention(d_model,
                                                 nhead,
                                                 dropout=dropout,
                                                 batch_first=batch_first)
        # Implementation of Feedexecute model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = nn.relu

    def execute(self,
                tgt: Var,
                memory: Var,
                tgt_mask: Optional[Var] = None,
                memory_mask: Optional[Var] = None,
                tgt_key_padding_mask: Optional[Var] = None,
                memory_key_padding_mask: Optional[Var] = None) -> Var:
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask,
                                   tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask,
                                    memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x +
                           self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask,
                                               memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Var, attn_mask: Optional[Var],
                  key_padding_mask: Optional[Var]) -> Var:
        x = self.self_attn(x,
                           x,
                           x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Var, mem: Var, attn_mask: Optional[Var],
                   key_padding_mask: Optional[Var]) -> Var:
        x = self.multihead_attn(x,
                                mem,
                                mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed execute block
    def _ff_block(self, x: Var) -> Var:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class TPN_Decoder(TransformerDecoder):

    def execute(self,
                tgt: Var,
                memory: Var,
                tgt_mask: Optional[Var] = None,
                memory_mask: Optional[Var] = None,
                tgt_key_padding_mask: Optional[Var] = None,
                memory_key_padding_mask: Optional[Var] = None,
                return_memory=False):
        if return_memory:
            output = tgt
            attns = []
            outputs = []
            ks = []
            vs = []
            for mod in self.layers:
                output, attn, k, v = mod(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    return_memory=True)
                attns.append(attn)
                outputs.append(output)
                ks.append(k)
                vs.append(v)

            if self.norm is not None:  # not do
                output = self.norm(output)

            return outputs, attns, ks, vs
        else:
            output = tgt
            attns = []
            outputs = []
            for mod in self.layers:
                output, attn = mod(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask)
                attns.append(attn)
                outputs.append(output)
            if self.norm is not None:  # not do
                output = self.norm(output)

            return outputs, attns


class TPN_DecoderLayer(TransformerDecoderLayer):

    def __init__(self, **kwargs):
        super(TPN_DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn
        del self.self_attn
        del self.norm1
        self.multihead_attn = Attention(kwargs['d_model'],
                                        num_heads=kwargs['nhead'],
                                        qkv_bias=True,
                                        attn_drop=0.1)

    def execute(self,
                tgt: Var,
                memory: Var,
                tgt_mask: Optional[Var] = None,
                memory_mask: Optional[Var] = None,
                tgt_key_padding_mask: Optional[Var] = None,
                memory_key_padding_mask: Optional[Var] = None,
                return_memory=False) -> Var:
        if return_memory:
            tgt0, attn, k, v = self.multihead_attn(tgt,
                                                   memory,
                                                   memory,
                                                   return_memory=True)
            tgt = tgt + self.dropout2(tgt0)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(
                self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
            return tgt, attn, k, v
        else:
            tgt0, attn = self.multihead_attn(tgt, memory, memory)
            tgt = tgt + self.dropout2(tgt0)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(
                self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
            return tgt


class RecoveryDecoder(nn.Module):

    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.decouple_q = nn.ModuleList()
        self.decouple_v = nn.ModuleList()
        for i in range(num_layers):
            self.decouple_q.append(
                TPN_DecoderLayer(d_model=dim,
                                 nhead=nhead,
                                 dim_feedforward=dim * 4))
            self.decouple_v.append(
                TPN_DecoderLayer(d_model=dim,
                                 nhead=nhead,
                                 dim_feedforward=dim * 4))
        self.linear_q_in = nn.Linear(dim, dim)
        self.linear_k_in = nn.Linear(dim, dim)
        self.linear_q_out = nn.Linear(dim, dim * 2)
        self.linear_k_out = nn.Linear(dim, dim)
        self.init_weights()

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def execute(self, q, lateral):
        q = self.linear_q_in(q)
        lateral = self.linear_k_in(lateral)
        for i in range(self.num_layers):
            q, lateral = self.decouple_q[i](q, lateral), self.decouple_v[i](
                lateral, q)
        q = self.linear_q_out(q)
        lateral = self.linear_k_out(lateral)
        return q, lateral
