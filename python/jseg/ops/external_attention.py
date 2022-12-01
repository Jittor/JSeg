from jittor import Module, nn
from jseg.bricks import ConvModule


class External_attention(Module):
    '''
    Arguments:
        c (int): The input and output channel number.
    '''
    def __init__(self, in_channels, channels, k=256):
        super(External_attention, self).__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.k = k

        self.conv1 = ConvModule(self.in_channels, self.channels, 1)

        self.linear_0 = ConvModule(self.channels, self.k, 1)

        self.linear_1 = ConvModule(self.k, self.channels, 1)

        self.conv2 = ConvModule(self.channels, self.channels, 1)

    def execute(self, x):
        x = self.conv1(x)
        idn = x
        b, c, h, w = x.size()
        x = self.linear_0(x)  # b, k, h, w
        x = x.view(b, self.k, h * w)  # b * k * n

        x = nn.softmax(x, dim=-1)  # b, k, n
        x = x / (1e-9 + x.sum(dim=1, keepdims=True))  # b, k, n

        x = x.view(b, self.k, h, w)
        x = self.linear_1(x)  # b, c, h, w

        x = x + idn
        x = self.conv2(x)
        return x
