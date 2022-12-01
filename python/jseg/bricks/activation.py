import jittor as jt
from jittor import nn

from jseg.utils.registry import ACTIVATION_LAYERS, build_from_cfg

for module in [
        nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ReLU6, nn.ELU, nn.Sigmoid, nn.Tanh
]:
    if module.__name__ == 'relu':
        ACTIVATION_LAYERS.register_module(name='ReLU', module=module)
    elif module.__name__ == 'relu6':
        ACTIVATION_LAYERS.register_module(name='ReLU6', module=module)
    else:
        ACTIVATION_LAYERS.register_module(module=module)


@ACTIVATION_LAYERS.register_module(name='Clip')
@ACTIVATION_LAYERS.register_module()
class Clamp(nn.Module):

    def __init__(self, min=-1., max=1.):
        super().__init__()
        self.min = min
        self.max = max

    def execute(self, x):
        return jt.clamp(x, min_v=self.min, max_v=self.max)


class GELU(nn.Module):

    def execute(self, input):
        return nn.gelu(input)


ACTIVATION_LAYERS.register_module(module=GELU)


def build_activation_layer(cfg):
    return build_from_cfg(cfg, ACTIVATION_LAYERS)
