# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.

from typing import Any, Dict, List, Optional, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from models.llama_attn import FlashAttention2, LlamaRMSNorm
logger = logging.get_logger(__name__)  


class BinaryAR(object):
    def __init__(self, num_tkns):
        self.num_tkns = num_tkns

    def sample(self, model, tokenizer, llm, prompts, guidance_scale=7.5, temp=1.0):
        bsz = len(prompts)
        y = prompts + [''] * bsz
        text_inputs = tokenizer(y, truncation=True, padding=True, max_length=128, return_tensors='pt', return_token_type_ids=False).to("cuda")
        llm_features = llm(**text_inputs, output_hidden_states=True).hidden_states[1:]
        for i, t in enumerate(range(self.num_tkns)):
            if i == 0:
                z_in = None
            else:
                z_in = torch.cat([z, z], 0)
            out = model(
                    hidden_states=z_in, 
                    llm_features=llm_features,
                    text_mask=text_inputs['attention_mask'],
                )
            out = out[:, -1:, :]
            out = out / temp
            (pred_cond, pred_uncond) = out.chunk(2, dim=0)
            x_0_logits = (1 + guidance_scale) * pred_cond - guidance_scale * pred_uncond
            x_0_logits = torch.sigmoid(x_0_logits)
            if temp != 0:
                z = torch.bernoulli(x_0_logits)
            else:
                z = (x_0_logits > 0.5) * 1.0
            
            if i > 0:
                z = torch.cat([z_in.chunk(2, dim=0)[0], z], dim=1)
            
        return z


@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, text_cond_dim, num_img_tkns, mlp_ratio=4.0):
        super().__init__()

        self.mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        self.input_layernorm = LlamaRMSNorm(dim, eps=1e-6)
        self.post_attention_layernorm = LlamaRMSNorm(dim, eps=1e-6)

        self.proj_cond = nn.Linear(text_cond_dim, dim)

        self.pos_emb = nn.Parameter(torch.zeros(1, num_img_tkns+1, dim))

        self.attn = FlashAttention2(
            hidden_size=dim,
            num_heads=num_attention_heads,
            eps=1e-6,
            is_causal=True,
        )


    def forward(
        self,
        hidden_states: torch.FloatTensor,
        text_cond=None,
        text_mask=None, 
    ):

        seq_length = hidden_states.shape[1]

        norm_hidden_states = self.input_layernorm(hidden_states)

        image_mask = torch.ones(norm_hidden_states.shape[0], norm_hidden_states.shape[1]).to(norm_hidden_states.device)
        joint_mask = torch.cat([text_mask, image_mask], dim=1)

        text_cond = self.proj_cond(text_cond)

        norm_hidden_states = norm_hidden_states + self.pos_emb[:, :norm_hidden_states.shape[1], :] # 1d learnable position embedding
        norm_hidden_states = torch.cat([text_cond, norm_hidden_states], dim=1)
        
        
        norm_hidden_states = self.attn(
            hidden_states=norm_hidden_states,
            attention_mask=joint_mask,
            query_length=seq_length
        )
        hidden_states = norm_hidden_states + hidden_states

        norm_hidden_states = self.post_attention_layernorm(hidden_states)
        norm_hidden_states = self.mlp(norm_hidden_states)
        hidden_states = norm_hidden_states + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


class Instella_AR_Model(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 64,
        num_layers: int = 16,
        attention_head_dim: int = 128,
        num_attention_heads: int = 16,
        num_img_tkns: int = 128,
        text_cond_dim: int = 2048, 
        axes_dims_rope: List[int] = [16, 56, 56],
    ):
        super().__init__()
        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.time_embed = TimestepEmbedding(in_channels=1, time_embed_dim=self.inner_dim)

        self.x_embedder = torch.nn.Linear(self.config.in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads = num_attention_heads , 
                    text_cond_dim = text_cond_dim, 
                    num_img_tkns = num_img_tkns,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = LlamaRMSNorm(self.inner_dim, 1e-6)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=True)

        self.split_token = nn.Parameter(torch.zeros(1, 1, self.inner_dim))
        self.split_token.data.normal_(0.0, 0.02)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        llm_features: list = None,
        text_mask = None,
        return_dict: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        

        assert len(llm_features) == self.config.num_layers
        if hidden_states is not None:
            hidden_states = self.x_embedder(hidden_states)

            split_tkn = self.split_token.repeat(hidden_states.shape[0], 1, 1)
            hidden_states = torch.cat([split_tkn, hidden_states], dim=1)
        else:
            bsz = llm_features[0].shape[0]
            hidden_states = self.split_token.repeat(bsz, 1, 1)

        for index_block, block in enumerate(self.transformer_blocks):
            llm_feature = llm_features[index_block]
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    llm_feature, 
                    text_mask,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    text_cond=llm_feature,
                    text_mask=text_mask
                )

        # hidden_states = hidden_states[:,:-1, :]
        hidden_states = self.norm_out(hidden_states)
        output = self.proj_out(hidden_states)


        if not return_dict:
            return output

        return Transformer2DModelOutput(sample=output)

if __name__ == '__main__':
    model = Instella_AR_Model()
    print(model)