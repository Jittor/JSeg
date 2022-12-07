from jittor import nn

from jseg.utils.general import add_prefix
from jseg.ops import resize
from jseg.utils.registry import MODELS, build_from_cfg, HEADS
from .encoder_decoder import EncoderDecoder


@MODELS.register_module()
class CascadeEncoderDecoder(EncoderDecoder):

    def __init__(self,
                 num_stages,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        self.num_stages = num_stages
        super(CascadeEncoderDecoder,
              self).__init__(backbone=backbone,
                             decode_head=decode_head,
                             neck=neck,
                             auxiliary_head=auxiliary_head,
                             train_cfg=train_cfg,
                             test_cfg=test_cfg,
                             pretrained=pretrained)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        assert isinstance(decode_head, list)
        assert len(decode_head) == self.num_stages
        self.decode_head = nn.ModuleList()
        for i in range(self.num_stages):
            self.decode_head.append(build_from_cfg(decode_head[i], HEADS))
        self.align_corners = self.decode_head[-1].align_corners
        self.num_classes = self.decode_head[-1].num_classes
        self.out_channels = self.decode_head[-1].out_channels

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self.decode_head[0].execute_test(x, img_metas, self.test_cfg)
        for i in range(1, self.num_stages):
            out = self.decode_head[i].execute_test(x, out, img_metas,
                                                   self.test_cfg)
        out = resize(input=out,
                     size=img.shape[2:],
                     mode='bilinear',
                     align_corners=self.align_corners)
        return out

    def _decode_head_execute_train(self, x, img_metas, gt_semantic_seg):
        """Run execute function and calculate loss for decode head in
        training."""
        losses = dict()

        loss_decode = self.decode_head[0].execute_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode_0'))

        for i in range(1, self.num_stages):
            # execute test again, maybe unnecessary for most methods.
            if i == 1:
                prev_outputs = self.decode_head[0].execute_test(
                    x, img_metas, self.test_cfg)
            else:
                prev_outputs = self.decode_head[i - 1].execute_test(
                    x, prev_outputs, img_metas, self.test_cfg)
            loss_decode = self.decode_head[i].execute_train(
                x, prev_outputs, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_decode, f'decode_{i}'))

        return losses
