import jittor as jt
from jittor import nn

from jseg.utils.registry import PIXEL_SAMPLERS
from .base_pixel_sampler import BasePixelSampler


@PIXEL_SAMPLERS.register_module()
class OHEMPixelSampler(BasePixelSampler):
    def __init__(self, context, thresh=None, min_kept=100000):
        super(OHEMPixelSampler, self).__init__()
        self.context = context
        assert min_kept > 1
        self.thresh = thresh
        self.min_kept = min_kept
        print('USE OHEM ', 'thresh:', thresh, 'min_kept', min_kept)

    def sample(self, seg_logit, seg_label):
        with jt.no_grad():
            assert seg_logit.shape[2:] == seg_label.shape[2:]
            assert seg_label.shape[1] == 1
            seg_label = seg_label.squeeze(1).long()
            batch_kept = self.min_kept * seg_label.size(0)
            valid_mask = seg_label != self.context.ignore_index
            seg_weight = jt.zeros(seg_label.size()).astype(seg_logit.dtype)
            valid_seg_weight = seg_weight[valid_mask]
            if self.thresh is not None:
                seg_prob = nn.softmax(seg_logit, dim=1)

                tmp_seg_label = seg_label.clone().unsqueeze(1)
                tmp_seg_label[tmp_seg_label == self.context.ignore_index] = 0
                seg_prob = jt.gather(seg_prob, 1, tmp_seg_label).squeeze(1)
                sort_indices, sort_prob = seg_prob[valid_mask].argsort()

                if sort_prob.numel() > 0:
                    min_threshold = sort_prob[min(batch_kept,
                                                  sort_prob.numel() - 1)]
                else:
                    min_threshold = 0.0
                threshold = max(min_threshold, self.thresh)
                valid_seg_weight[seg_prob[valid_mask] < threshold] = 1.
            else:
                if not isinstance(self.context.loss_decode, nn.ModuleList):
                    losses_decode = [self.context.loss_decode]
                else:
                    losses_decode = self.context.loss_decode
                losses = 0.0
                for loss_module in losses_decode:
                    losses += loss_module(
                        seg_logit,
                        seg_label,
                        weight=None,
                        ignore_index=self.context.ignore_index,
                        reduction_override='none')

                # faster than topk according to https://github.com/pytorch/pytorch/issues/22812  # noqa
                sort_indices, _ = losses[valid_mask].argsort(descending=True)
                valid_seg_weight[sort_indices[:batch_kept]] = 1.

            seg_weight[valid_mask] = valid_seg_weight

            return seg_weight
