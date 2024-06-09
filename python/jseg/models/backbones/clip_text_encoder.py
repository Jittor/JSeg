import jittor as jt
from jittor import nn
from jseg.utils.registry import BACKBONES

from jseg.ops.cliprc_ops import Transformer, LayerNorm


@BACKBONES.register_module()
class CLIPTextEncoder(nn.Module):

    def __init__(self,
                 context_length=77,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=1024,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.context_length = context_length
        self.transformer = Transformer(width=transformer_width,
                                       layers=transformer_layers,
                                       heads=transformer_heads,
                                       attn_mask=self.build_attention_mask())

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = jt.empty(
            (self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = jt.empty((transformer_width, embed_dim))

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = jt.load(pretrained)

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    state_dict[k] = checkpoint[k]

                if k == 'positional_embedding' or k == 'text_projection' or k.startswith(
                        'token_embedding') or k.startswith('ln_final'):
                    if k == 'positional_embedding' and checkpoint[k].size(
                            0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        print('positional_embedding is tuncated from 77 to',
                              self.context_length)
                    state_dict[k] = checkpoint[k]

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in text encoder')

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = jt.empty((self.context_length, self.context_length))
        mask.fill_(float("-inf"))
        mask = jt.triu_(mask, 1)  # zero out the lower diagonal
        return mask

    def execute(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[jt.arange(x.shape[0]),
              text.argmax(dim=-1)] @ self.text_projection
        return x
