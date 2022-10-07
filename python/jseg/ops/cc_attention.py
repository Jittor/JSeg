import jittor as jt
from jittor import nn

from .scale import Scale


def NEG_INF_DIAG(n):
    return jt.diag(jt.Var(float('-inf')).repeat(n), 0)


class CrissCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = Scale(0.)
        self.in_channels = in_channels

    def execute(self, x):
        B, C, H, W = x.size()
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        energy_H = jt.linalg.einsum('bchw,bciw->bwhi', query,
                                    key) + NEG_INF_DIAG(H)
        energy_H = energy_H.transpose(1, 2)
        energy_W = jt.linalg.einsum('bchw,bchj->bhwj', query, key)
        attn = nn.softmax(jt.concat([energy_H, energy_W], dim=-1),
                          dim=-1)  # [B,H,W,(H+W)]
        out = jt.linalg.einsum('bciw,bhwi->bchw', value, attn[..., :H])
        out += jt.linalg.einsum('bchj,bhwj->bchw', value, attn[..., H:])

        out = self.gamma(out) + x

        return out

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(in_channels={self.in_channels})'
        return s
