# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.

# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Dict, List, Optional, Union
import math
import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero
from diffusers.utils import logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from models.llama_attn import FlashAttention2
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class BinaryDiffusion(object):
    def __init__(self, num_tkns, codebook_size, weight_dtype):
        self.num_tkns = num_tkns
        self.codebook_size = codebook_size
        self.weight_dtype = weight_dtype

    def add_noise(self, latents, t):
        x_t = latents * (1 - t) + t * (0.5*torch.ones_like(latents))
        x_t = torch.bernoulli(x_t)
        return x_t

    
    def sample(self, model, tokenizer, llm, prompts, num_sampling_steps=20, guidance_scale=7.5, temp=1.0, rho=1.0, protocal='protocal_1'):
        bsz = len(prompts)
        y = prompts + [''] * bsz
        z = torch.bernoulli(0.5*torch.ones(bsz, self.num_tkns, self.codebook_size, device='cuda')).to(self.weight_dtype)
        text_inputs = tokenizer(y, truncation=True, padding=True, max_length=128, return_tensors='pt', return_token_type_ids=False).to("cuda")
        llm_features = llm(**text_inputs, output_hidden_states=True).hidden_states[1:]
        timesteps_ori = [(1.0 - i / num_sampling_steps)**rho  for i in range(num_sampling_steps)]
        timesteps = [torch.tensor([t] * (z.shape[0])).to('cuda', dtype=self.weight_dtype) for t in timesteps_ori]
        for i, t in enumerate(timesteps):
            z_in = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)

            out = model(
                    hidden_states=z_in, 
                    llm_features=llm_features,
                    text_mask=text_inputs['attention_mask'],
                    timestep=t,
                )
            out = out / temp
            (pred_cond, pred_uncond) = out.chunk(2, dim=0)

            x_0_logits = (1 + guidance_scale) * pred_cond - guidance_scale * pred_uncond
            x_0_logits = torch.sigmoid(x_0_logits)
            x_0_logits =  z * (1 - x_0_logits) + (1 - z) * x_0_logits
            if i < len(timesteps) - 1:
                if protocal == 'protocal_1':
                    timepoints = timesteps[i + 1]
                    z = x_0_logits * (1 - timepoints[:, None, None]) + 0.5*torch.ones_like(z) * timepoints[:, None, None]
                    z = torch.bernoulli(z)
                elif protocal == 'protocal_2':
                    x_0_logits = torch.cat([x_0_logits.unsqueeze(-1), (1-x_0_logits).unsqueeze(-1)], dim=-1)
                    x_t_logits = torch.cat([z.unsqueeze(-1), (1-z).unsqueeze(-1)], dim=-1)

                    t = timesteps[i][:, None, None, None]
                    tm1 =  timesteps[i + 1][:, None, None, None]
                    p_EV_qxtmin_x0 = x_0_logits * (1.0-tm1) + 0.5 * tm1 * torch.ones_like(x_0_logits)

                    step_size = t - tm1
                    q_one_step = x_t_logits - step_size * (x_t_logits - 0.5)

                    unnormed_probs = p_EV_qxtmin_x0 * q_one_step
                    unnormed_probs = unnormed_probs / unnormed_probs.sum(-1, keepdims=True)

                    z = unnormed_probs[...,0]
                    z = torch.bernoulli(z)
                else:
                    raise NotImplementedError
                
            else:
                z = (x_0_logits > 0.5) * 1.0
        return z
        


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)

    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


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
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = AdaLayerNormZero(dim)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.proj_cond = nn.Linear(text_cond_dim, dim)

        self.pos_emb = nn.Parameter(torch.zeros(1, num_img_tkns, dim))

        self.attn = FlashAttention2(
            hidden_size=dim,
            num_heads=num_attention_heads,
            eps=1e-6,
            is_causal=False,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)


    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        text_cond=None,
        text_mask=None, 
    ):

        seq_length = hidden_states.shape[1]

        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        image_mask = torch.ones(norm_hidden_states.shape[0], norm_hidden_states.shape[1]).to(norm_hidden_states.device)
        # joint_mask = torch.cat([image_mask, text_mask], dim=1)
        joint_mask = torch.cat([text_mask, image_mask], dim=1)

        num_img_tkns = norm_hidden_states.shape[1]
        text_cond = self.proj_cond(text_cond)

        norm_hidden_states = norm_hidden_states + self.pos_emb # 1d learnable position embedding
        # norm_hidden_states = torch.cat([norm_hidden_states, text_cond], dim=1)
        norm_hidden_states = torch.cat([text_cond, norm_hidden_states], dim=1)

        
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            attention_mask=joint_mask,
            query_length=seq_length
        )
        attn_output = attn_output[:, -num_img_tkns:, ...]
        attn_output = gate_msa.unsqueeze(1) * attn_output

        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]


        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states
    
    
class Instella_Binary_Diff(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
    """

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

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=True)

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        llm_features: list = None,
        text_mask = None,
        timestep: torch.LongTensor = None, # range [0-1]
        return_dict: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    
        assert len(llm_features) == self.config.num_layers
        hidden_states = self.x_embedder(hidden_states)
        ### from latent dim to DiT dim
        timestep = timestep.to(hidden_states.dtype) 
        temb = self.time_embed(timestep.unsqueeze(-1))

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

                # ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    llm_feature, 
                    text_mask,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    text_cond=llm_feature,
                    text_mask=text_mask
                )


        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)


        if not return_dict:
            return output

        return Transformer2DModelOutput(sample=output)

if __name__ == '__main__':
    model = Instella_Binary_Diff()
    print(model)