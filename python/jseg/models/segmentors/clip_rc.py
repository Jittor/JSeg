import jittor as jt
from jittor import nn
import numpy as np

from jseg.ops import resize
from jseg.utils.general import add_prefix
from jseg.utils.registry import MODELS, build_from_cfg, BACKBONES, HEADS
from .encoder_decoder import EncoderDecoder

from jseg.utils.tokenizer import tokenize


@MODELS.register_module()
class CLIPRC(EncoderDecoder):
    """Encoder Decoder segmentors.
    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 text_encoder,
                 pretrained_text,
                 class_names,
                 base_class,
                 novel_class,
                 both_class,
                 multi_prompts=False,
                 self_training=False,
                 ft_backbone=False,
                 exclude_key=None,
                 load_text_embedding=None,
                 **args):
        super(CLIPRC, self).__init__(**args)

        if pretrained_text is not None:
            assert text_encoder.get('pretrained') is None, \
                'both text encoder and segmentor set pretrained weight'
            text_encoder.pretrained = pretrained_text

        self.text_encoder = build_from_cfg(text_encoder, BACKBONES)

        self.class_names = class_names

        self.base_class = np.asarray(base_class)
        self.novel_class = np.asarray(novel_class)
        self.both_class = np.asarray(both_class)
        self.self_training = self_training
        self.multi_prompts = multi_prompts
        self.load_text_embedding = load_text_embedding

        if not self.load_text_embedding:
            if not self.multi_prompts:
                self.texts = jt.concat(
                    [tokenize(f"a photo of a {c}") for c in self.class_names])
            else:
                self.texts = self._get_multi_prompts(self.class_names)

        if len(self.base_class) != len(self.both_class):  # zero-shot setting
            if not self_training:
                self._visiable_mask(self.base_class)
            else:
                self._visiable_mask_st(self.base_class)
                self._st_mask(self.novel_class)

        if self.training:
            self._freeze_stages(self.text_encoder)
            if ft_backbone is False:
                self._freeze_stages(self.backbone, exclude_key=exclude_key)
            if jt.rank == 0:
                print('--------------------------------------')
            for n, m in self.named_parameters():
                if m.requires_grad:
                    if jt.rank == 0:
                        print('Finetune layer in segmentor:', n)
        else:
            self.text_encoder.eval()
            self.backbone.eval()

    def _freeze_stages(self, model, exclude_key=None):
        """Freeze stages param and norm stats."""
        for n, m in model.named_parameters():
            if exclude_key:
                if isinstance(exclude_key, str):
                    if not exclude_key in n:
                        m.requires_grad = False
                elif isinstance(exclude_key, list):
                    count = 0
                    for i in range(len(exclude_key)):
                        i_layer = str(exclude_key[i])
                        if i_layer in n:
                            count += 1
                    if count == 0:
                        m.requires_grad = False
                    elif count > 0:
                        if jt.rank == 0:
                            print('Finetune layer in backbone:', n)
                else:
                    assert AttributeError(
                        "Dont support the type of exclude_key!")
            else:
                m.requires_grad = False

    def _visiable_mask(self, seen_classes):
        seen_map = np.array([-1] * 256)
        seen_map[255] = 255
        for i, n in enumerate(list(seen_classes)):
            seen_map[n] = i
        self.visibility_seen_mask = seen_map.copy()
        if jt.rank == 0:
            print('Making visible mask for zero-shot setting:',
                  self.visibility_seen_mask)

    def _visiable_mask_st(self, seen_classes):
        seen_map = np.array([-1] * 256)
        seen_map[255] = 255
        for i, n in enumerate(list(seen_classes)):
            seen_map[n] = n
        seen_map[200] = 200  # pixels of padding will be excluded
        self.visibility_seen_mask = seen_map.copy()
        if jt.rank == 0:
            print(
                'Making visible mask for zero-shot setting in self_traning stage:',
                self.visibility_seen_mask)

    def _st_mask(self, novel_classes):
        st_mask = np.array([255] * 256)
        st_mask[255] = 255
        for i, n in enumerate(list(novel_classes)):
            st_mask[n] = n
        self.st_mask = st_mask.copy()
        if jt.rank == 0:
            print(
                'Making st mask for zero-shot setting in self_traning stage:',
                self.st_mask)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = build_from_cfg(decode_head, HEADS)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _decode_head_execute_train(self, feat, img_metas, gt_semantic_seg):
        """Run execute function and calculate loss for decode head in
        training."""
        if self.training:
            if len(self.base_class) != len(self.both_class):  # zero setting
                gt_semantic_seg = jt.Var(self.visibility_seen_mask).type_as(
                    gt_semantic_seg)[gt_semantic_seg]

        losses = dict()
        if self.self_training:
            loss_decode = self.decode_head.execute_train(
                feat, img_metas, gt_semantic_seg, self.train_cfg,
                self.self_training, self.st_mask)
        else:
            loss_decode = self.decode_head.execute_train(
                feat, img_metas, gt_semantic_seg, self.train_cfg,
                self.self_training)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def text_embedding(self, texts, img):
        text_embeddings = self.text_encoder(texts)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1,
                                                                 keepdim=True)
        return text_embeddings

    def extract_feat(self, img):
        """Extract features from images."""
        visual_feat = self.backbone(img)
        return visual_feat

    def execute_train(self, img, img_metas, gt_semantic_seg):
        # load pth from /home/zy/projects/CLIP-RC/img, img_metas, gt_semantic_seg
        # img = jt.load('/home/zy/projects/CLIP-RC/img.pth')
        # img_metas = jt.load('/home/zy/projects/CLIP-RC/img_metas.pth')
        # gt_semantic_seg = jt.load('/home/zy/projects/CLIP-RC/gt_semantic_seg.pth')
        visual_feat = self.extract_feat(img)
        if self.load_text_embedding:
            text_feat = np.load(self.load_text_embedding)
            text_feat = jt.Var(text_feat)
        else:
            if not self.multi_prompts:
                text_feat = self.text_embedding(self.texts, img)
            else:
                assert AttributeError("preparing the multi embeddings")

        if not self.self_training:
            text_feat = text_feat[self.base_class, :]

        feat = []
        feat.append(visual_feat)
        feat.append(text_feat)

        losses = dict()
        loss_decode = self._decode_head_execute_train(feat, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)
        return losses

    def encode_decode(self, img, img_metas):
        visual_feat = self.extract_feat(img)

        if self.load_text_embedding:
            text_feat = np.load(self.load_text_embedding)
            text_feat = jt.Var(text_feat)
        else:
            if not self.multi_prompts:
                text_feat = self.text_embedding(self.texts, img)
            else:
                num_cls, num_prompts, _ = self.texts.size()
                text_feat = self.text_embedding(
                    self.texts.reshape(num_cls * num_prompts, -1), img)
                text_feat = text_feat.reshape(num_cls, num_prompts,
                                              -1).mean(dim=1)
                text_feat /= text_feat.norm(dim=-1).unsqueeze(1)

        feat = []
        feat.append(visual_feat)
        feat.append(text_feat)

        out = self._decode_head_execute_test(feat, img_metas,
                                             self.self_training)
        out = resize(input=out,
                     size=img.shape[2:],
                     mode='bilinear',
                     align_corners=self.align_corners)
        return out

    def _decode_head_execute_test(self, x, img_metas, self_training):
        """Run execute function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.execute_test(x, img_metas, self.test_cfg,
                                                   self_training)
        return seg_logits

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = len(self.both_class)
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
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
            preds = resize(preds,
                           size=img_meta['ori_shape'][:2],
                           mode='bilinear',
                           align_corners=self.align_corners,
                           warning=False)
        return preds

    def simple_test(self, img, img_meta, rescale=True):
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)[0]
        # unravel batch dim
        seg_pred = list(seg_pred.numpy())
        return seg_pred
