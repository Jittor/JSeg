import jittor as jt
from jseg.ops.non_local import NonLocal2d

from jseg.utils.registry import HEADS
from .fcn_head import FCNHead


@HEADS.register_module()
class NLHead(FCNHead):
    def __init__(self,
                 reduction=2,
                 use_scale=True,
                 mode='embedded_gaussian',
                 **kwargs):
        super(NLHead, self).__init__(num_convs=2, **kwargs)
        self.reduction = reduction
        self.use_scale = use_scale
        self.mode = mode
        self.nl_block = NonLocal2d(in_channels=self.channels,
                                   reduction=self.reduction,
                                   use_scale=self.use_scale,
                                   conv_cfg=self.conv_cfg,
                                   norm_cfg=self.norm_cfg,
                                   mode=self.mode)

    def execute(self, inputs):
        """execute function."""
        x = self._transform_inputs(inputs)
        output = self.convs[0](x)
        output = self.nl_block(output)
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(jt.concat([x, output], dim=1))
        output = self.cls_seg(output)
        return output
