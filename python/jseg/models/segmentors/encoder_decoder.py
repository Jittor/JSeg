import jittor as jt
from jittor import nn

from jseg.utils.general import add_prefix
from jseg.ops import resize
from jseg.utils.registry import MODELS, build_from_cfg, BACKBONES, NECKS, HEADS
from .base import BaseSegmentor


@MODELS.register_module()
class EncoderDecoder(BaseSegmentor):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EncoderDecoder, self).__init__()
        self.backbone = build_from_cfg(backbone, BACKBONES)
        if neck is not None:
            self.neck = build_from_cfg(backbone, NECKS)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        self.decode_head = build_from_cfg(decode_head, HEADS)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head):
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(build_from_cfg(head_cfg, HEADS))
            else:
                self.auxiliary_head = build_from_cfg(auxiliary_head, HEADS)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(EncoderDecoder, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_execute_test(x, img_metas)
        out = resize(input=out,
                     size=img.shape[2:],
                     mode='bilinear',
                     align_corners=self.align_corners)
        return out

    def _decode_head_execute_train(self, x, img_metas, gt_semantic_seg):
        losses = dict()
        loss_decode = self.decode_head.execute_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_execute_test(self, x, img_metas):
        seg_logits = self.decode_head.execute_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_execute_train(self, x, img_metas, gt_semantic_seg):
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.execute_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.execute_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def execute_dummy(self, img):
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def execute_train(self, img, img_metas, gt_semantic_seg):
        x = self.extract_feat(img)

        losses = dict()

        loss_decode = self._decode_head_execute_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_execute_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = jt.zeros((batch_size, out_channels, h_img, w_img))
        count_mat = jt.zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += nn.pad(crop_seg_logit,
                                (int(x1), int(preds.shape[3] - x2), int(y1),
                                 int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        if rescale:
            resize_shape = img_meta['img_shape'][:2]
            preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
            preds = resize(preds,
                           size=img_meta['ori_shape'][:2],
                           mode='bilinear',
                           align_corners=self.align_corners,
                           warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            resize_shape = img_meta['img_shape'][:2]
            seg_logit = seg_logit[:, :, :resize_shape[0], :resize_shape[1]]
            size = img_meta['ori_shape'][:2]
            seg_logit = resize(seg_logit,
                               size=size,
                               mode='bilinear',
                               align_corners=self.align_corners,
                               warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        assert self.test_cfg.mode in ['slide', 'whole']
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = nn.softmax(seg_logit, dim=1)
        flip = img_meta['flip']
        if flip:
            flip_direction = img_meta['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = jt.flip(output, dim=(3, ))
            elif flip_direction == 'vertical':
                output = jt.flip(output, dim=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        seg_logit = self.inference(img, img_meta, rescale)
        if self.out_channels == 1:
            seg_pred = (seg_logit >
                        self.decode_head.threshold).to(seg_logit).squeeze(1)
        else:
            seg_pred = seg_logit.argmax(dim=1)[0]
        # unravel batch dim
        seg_pred = list(seg_pred.numpy())
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        if self.out_channels == 1:
            seg_pred = (seg_logit >
                        self.decode_head.threshold).to(seg_logit).squeeze(1)
        else:
            seg_pred = seg_logit.argmax(dim=1)[0]
        # unravel batch dim
        seg_pred = list(seg_pred.numpy())
        return seg_pred
