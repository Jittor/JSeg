import jittor as jt
from .fcn_head import FCNHead
from jseg.utils.registry import HEADS

try:
    from jseg.ops import CrissCrossAttention
except ModuleNotFoundError:
    CrissCrossAttention = None


@HEADS.register_module()
class CCHead(FCNHead):
    def __init__(self, recurrence=2, **kwargs):
        if CrissCrossAttention is None:
            raise RuntimeError('Please install mmcv-full for '
                               'CrissCrossAttention ops')
        super(CCHead, self).__init__(num_convs=2, **kwargs)
        self.recurrence = recurrence
        self.cca = CrissCrossAttention(self.channels)

    def execute(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs[0](x)
        for _ in range(self.recurrence):
            output = self.cca(output)
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(jt.concat([x, output], dim=1))
        output = self.cls_seg(output)
        return output
