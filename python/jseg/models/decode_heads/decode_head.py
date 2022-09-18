from abc import ABCMeta, abstractmethod

import jittor as jt
from jittor import nn

from jseg.ops import resize
from jseg.utils.registry import build_from_cfg, LOSSES
from ..losses import accuracy


class BaseDecodeHead(nn.Module, metaclass=ABCMeta):
    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(type='CrossEntropyLoss',
                                  use_sigmoid=False,
                                  loss_weight=1.0),
                 decoder_params=None,
                 ignore_index=255,
                 sampler=None,
                 align_corners=False):
        super(BaseDecodeHead, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_from_cfg(loss_decode, LOSSES)
        self.ignore_index = ignore_index
        self.align_corners = align_corners

        if sampler is not None:
            # TODO
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)
        else:
            self.dropout = None

    def extra_repr(self):
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def init_weights(self):
        pass

    def _transform_inputs(self, inputs):
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(input=x,
                       size=inputs[0].shape[2:],
                       mode='bilinear',
                       align_corners=self.align_corners) for x in inputs
            ]
            inputs = jt.concat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @abstractmethod
    def execute(self, inputs):
        pass

    def execute_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits = self.execute(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def execute_test(self, inputs, img_metas, test_cfg):
        return self.execute(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def losses(self, seg_logit, seg_label):
        loss = dict()
        seg_logit = resize(input=seg_logit,
                           size=seg_label.shape[2:],
                           mode='bilinear',
                           align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(seg_logit,
                                            seg_label,
                                            weight=seg_weight,
                                            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss
