from abc import ABCMeta, abstractmethod

from .decode_head import BaseDecodeHead


class BaseCascadeDecodeHead(BaseDecodeHead, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super(BaseCascadeDecodeHead, self).__init__(*args, **kwargs)

    @abstractmethod
    def execute(self, inputs, prev_output):
        pass

    def execute_train(self, inputs, prev_output, img_metas, gt_semantic_seg,
                      train_cfg):
        seg_logits = self.execute(inputs, prev_output)
        losses = self.losses(seg_logits, gt_semantic_seg)

        return losses

    def execute_test(self, inputs, prev_output, img_metas, test_cfg):
        return self.execute(inputs, prev_output)
