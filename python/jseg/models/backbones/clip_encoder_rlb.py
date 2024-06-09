import math
import jittor as jt
from jittor import nn
from jseg.utils.registry import BACKBONES
from functools import reduce
from operator import mul

from jseg.ops.cliprc_ops import Transformer, LayerNorm


@BACKBONES.register_module()
class CLIPVisionTransformerWithRLB(nn.Module):

    def __init__(self,
                 input_resolution=224,
                 patch_size=32,
                 width=768,
                 layers=12,
                 heads=12,
                 output_dim=512,
                 drop_path_rate=0.0,
                 out_indices=[3, 5, 7, 11],
                 pretrained=None,
                 get_embeddings=False,
                 num_tokens=20,
                 prompt_dim=512,
                 total_d_layer=11,
                 region_level_bridge_size=16,
                 **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               bias=False)

        scale = width**-0.5
        self.class_embedding = scale * jt.randn(width)
        self.positional_embedding = scale * jt.randn(
            (input_resolution // patch_size)**2 + 1, width)
        self.spatial_size = input_resolution // patch_size
        self.ln_pre = LayerNorm(width)
        self.get_embeddings = get_embeddings
        self.num_layers = layers

        ## Setting of visual prompt tuning
        self.num_tokens = num_tokens
        self.prompt_dim = prompt_dim
        self.total_d_layer = total_d_layer

        self.region_level_bridge_size = region_level_bridge_size

        self.region_level_bridge_hw = int(
            math.sqrt(self.region_level_bridge_size))
        self.region_level_bridge = jt.zeros(self.region_level_bridge_size,
                                            prompt_dim)

        visual_mask = self.gen_attention_mask(stride=self.spatial_size //
                                              self.region_level_bridge_hw)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            drop_path_rate=drop_path_rate,
            attn_mask=visual_mask,
        )

        self.out_indices = out_indices

        if get_embeddings:
            self.ln_post = LayerNorm(width)
            self.proj = scale * jt.randn(width, output_dim)

        # Add the prompt parameters
        self._init_prompt(patch_size, self.num_tokens, self.prompt_dim,
                          self.total_d_layer)

    def gen_attention_mask(self, stride):
        # cls, vpt, image feature, region level bridge
        att_size = 1 + self.num_tokens + self.spatial_size**2 + self.region_level_bridge_size
        rlb_index = -self.region_level_bridge_size
        visual_mask = jt.zeros((att_size, att_size),
                               dtype=jt.float32).stop_grad()
        visual_mask[:, rlb_index:] = float("-inf")
        visual_mask[rlb_index:, :] = float("-inf")
        # gen att_size * att_size index
        for i in range(self.region_level_bridge_hw):
            for j in range(self.region_level_bridge_hw):
                tmp_mask = jt.zeros(
                    (self.spatial_size, self.spatial_size)).stop_grad()
                tmp_mask[i * stride:(i + 1) * stride,
                         j * stride:(j + 1) * stride] = 1

                idx = tmp_mask.flatten().nonzero() + self.num_tokens + 1
                visual_mask[rlb_index + i * self.region_level_bridge_hw + j,
                            idx] = 0

                visual_mask[idx, rlb_index + i * self.region_level_bridge_hw +
                            j] = 0

        for i in range(self.region_level_bridge_size):
            visual_mask[rlb_index + i, rlb_index + i] = 0

        # import cv2
        # cv2.imwrite('visual_mask.png', (visual_mask.numpy() + 1) * 255)

        return visual_mask

    def _init_prompt(self, patch, num_tokens, prompt_dim, total_d_layer):
        patch_size = []
        patch_size.append(patch)
        patch_size.append(patch)
        val = math.sqrt(
            6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

        if total_d_layer >= 0:
            self.prompt_embeddings = jt.zeros(1, num_tokens, prompt_dim)
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings, -val, val)

            if total_d_layer > 0:  # noqa
                self.deep_prompt_embeddings = jt.zeros(total_d_layer,
                                                       num_tokens, prompt_dim)
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings, -val, val)

            self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight,
                                    a=0,
                                    mode='fan_out')
            self.prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
            self.prompt_dropout = nn.Dropout(0.1)

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = jt.load(pretrained)

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            if 'positional_embedding' in state_dict.keys():
                if self.positional_embedding.shape != state_dict[
                        'positional_embedding'].shape:
                    # (1025, 768)                      (197, 768)
                    print(
                        f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} to {self.positional_embedding.shape}'
                    )
                    cls_pos = jt.Var(
                        state_dict["positional_embedding"][0:1, :])

                    spatial_pos = nn.interpolate(
                        jt.Var(state_dict["positional_embedding"])[
                            1:,
                        ].reshape(1, 14, 14, 768).permute(0, 3, 1, 2),
                        size=(self.spatial_size, self.spatial_size),
                        mode='bilinear')
                    spatial_pos = spatial_pos.reshape(
                        768,
                        self.spatial_size * self.spatial_size).permute(1, 0)
                    positional_embedding = jt.concat([cls_pos, spatial_pos],
                                                     dim=0)
                    state_dict['positional_embedding'] = positional_embedding
                    assert self.positional_embedding.shape == state_dict[
                        'positional_embedding'].shape

            self.load_state_dict(state_dict)

            # init self.region_level_bridge by cls_token
            region_level_bridge_init = jt.Var(
                state_dict['class_embedding']).repeat(
                    self.region_level_bridge_size, 1)

            self.region_level_bridge.data = region_level_bridge_init + cls_pos.repeat(
                self.region_level_bridge_size, 1)

    def execute(self, x: jt.Var):
        x = self.conv1(x)
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = jt.concat([
            self.class_embedding.to(x.dtype) +
            jt.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype), x
        ],
                      dim=1)

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0, :] + self.class_embedding.to(x.dtype)
        spatial_pos = nn.interpolate(pos[
            1:,
        ].reshape(1, self.spatial_size, self.spatial_size,
                  C).permute(0, 3, 1, 2),
                                     size=(H, W),
                                     mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H * W).permute(0, 2, 1)
        pos = jt.concat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos

        # x concat self.region_level_bridge
        x = jt.concat(
            [x, self.region_level_bridge.unsqueeze(0).expand(B, -1, -1)],
            dim=1)
        x = self.ln_pre(x)

        if self.total_d_layer >= 0:
            # concat prompt
            x = jt.concat((x[:, :1, :],
                           self.prompt_dropout(
                               self.prompt_proj(self.prompt_embeddings).expand(
                                   B, -1, -1)), x[:, 1:, :]),
                          dim=1)

        x = x.permute(1, 0, 2)

        features = []
        outs = []
        if self.total_d_layer == 0:  #shallow
            for i, blk in enumerate(self.transformer.resblocks):
                x = blk(x)
                if len(self.out_indices) > 1:
                    if i in self.out_indices:
                        xp = x.permute(1, 0,
                                       2)[:, 1 + self.num_tokens:, :].permute(
                                           0, 2, 1).reshape(B, -1, H, W)
                        features.append(xp.contiguous())
        elif self.total_d_layer > 0:  # deep
            x, features = self.execute_deep_prompt(x, features, H, W)

        if self.get_embeddings:
            x = x.permute(1, 0, 2)
            x = self.ln_post(x)
            x = x @ self.proj

            global_embedding = x[:, 0]
            visual_embedding = x[:, 1 + self.num_tokens:-self.
                                 region_level_bridge_size].reshape(
                                     B, H, W, -1).permute(0, 3, 1, 2)
            region_level_bridge = x[:,
                                    -self.region_level_bridge_size:].reshape(
                                        B, self.region_level_bridge_hw,
                                        self.region_level_bridge_hw,
                                        -1).permute(0, 3, 1, 2)

            if len(self.out_indices) == 1:
                visual_embedding = visual_embedding / visual_embedding.norm(
                    dim=1, keepdim=True)
                features.append(visual_embedding)

            outs.append(tuple(features))
            global_embedding = global_embedding / global_embedding.norm(
                dim=1, keepdim=True)
            outs.append(global_embedding)
            region_level_bridge = region_level_bridge / region_level_bridge.norm(
                dim=1, keepdim=True)
            outs.append(region_level_bridge)
        return outs

    def execute_deep_prompt(self,
                            embedding_output,
                            features,
                            H,
                            W,
                            out_last=False):
        B = embedding_output.shape[1]

        for i in range(self.num_layers):
            if i == 0:
                hidden_states = self.transformer.resblocks[i](embedding_output)
            elif i <= self.deep_prompt_embeddings.shape[0]:
                deep_prompt_emb = self.prompt_dropout(
                    self.prompt_proj(
                        self.deep_prompt_embeddings[i -
                                                    1]).unsqueeze(0).expand(
                                                        B, -1,
                                                        -1)).permute(1, 0, 2)
                hidden_states = jt.concat(
                    (hidden_states[:1, :, :], deep_prompt_emb,
                     hidden_states[(1 + self.num_tokens):, :, :]),
                    dim=0)
                hidden_states = self.transformer.resblocks[i](hidden_states)
            else:
                hidden_states = jt.concat(
                    (hidden_states[:1, :, :],
                     hidden_states[(1 + self.num_tokens):, :, :]),
                    dim=0)
                hidden_states = self.transformer.resblocks[i](hidden_states)

            if len(self.out_indices) > 1:
                if i in self.out_indices:
                    xp = hidden_states.permute(
                        1, 0, 2)[:,
                                 (1 + self.num_tokens
                                  ):-self.region_level_bridge_size, :].permute(
                                      0, 2, 1).reshape(B, -1, H, W)
                    features.append(xp.contiguous())

            if i == (self.num_layers - 2):
                before_last_feats = self.prompt_norm(hidden_states)

        encoded = jt.concat(
            (self.prompt_norm(
                hidden_states[:-self.region_level_bridge_size, :, :]),
             hidden_states[-self.region_level_bridge_size:, :, :]),
            dim=0)
        if out_last:
            return before_last_feats
        else:
            return encoded, features
