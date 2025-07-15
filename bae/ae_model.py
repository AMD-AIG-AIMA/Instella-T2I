# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.

from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from bae.llama_attn import LlamaFlashAttention2, LlamaMLP, LlamaRMSNorm
from diffusers.models.embeddings import FluxPosEmbed
from einops import rearrange
from bae.bnl import BinaryQuantizer
from bae.resize_layers import DCUpBlock2d, DCDownBlock2d

logger = logging.get_logger(__name__) 


@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):


    def __init__(self, dim, num_attention_heads, mlp_ratio=4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = LlamaMLP(dim, self.mlp_hidden_dim)
        self.input_layernorm = LlamaRMSNorm(dim, eps=1e-6)
        self.post_attention_layernorm = LlamaRMSNorm(dim, eps=1e-6)


        self.attn = LlamaFlashAttention2(
            hidden_size=dim,
            num_heads=num_attention_heads,
            eps=1e-6,
            is_causal=False,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        image_rotary_emb=None,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states = self.attn(
            hidden_states=hidden_states,
            position_embeddings=image_rotary_emb,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


class AE_Encoder(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
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
        in_channels: int = 3,
        num_layers: int = 12,
        attention_head_dim: int = 64,
        num_attention_heads: int = 12,
        patch_size: int = 16,
        num_latent_tkns: int = 128,
        downsample_idx: List[int] = [],
        axes_dims_rope: List[int] = [8, 28, 28],
    ):
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim

        # self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)
        self.patch_embedder = nn.Conv2d(in_channels, self.inner_dim, kernel_size=patch_size, stride=patch_size)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads = num_attention_heads , 
                )
                for _ in range(num_layers)
            ]
        )

        self.downsample_idx = downsample_idx
        if len(downsample_idx)>=1:
            self.down_layers = nn.ModuleList([DCDownBlock2d(dim=self.inner_dim) for _ in range(len(downsample_idx))])

        self.norm_out = LlamaRMSNorm(self.inner_dim, 1e-6)
        self.proj_out = nn.Linear(self.inner_dim, self.inner_dim, bias=True)

        self.latent_tokens = nn.Parameter(torch.zeros(1, num_latent_tkns, self.inner_dim))
        self.latent_tokens.data.normal_(0.0, 0.02)

        self.num_latent_tkns = num_latent_tkns

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        self.gradient_checkpointing = False
        self.scale = None

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value


    def get_pos_embed(self, hidden_states, H, W, scale=None):
        dtype = self.patch_embedder.weight.dtype
        device = hidden_states.device

        txt_ids = torch.zeros(self.num_latent_tkns, 3)
        txt_ids[:, 0] = txt_ids[:, 0] + torch.arange(self.num_latent_tkns)
        txt_ids = txt_ids.to(device=device, dtype=dtype)

        img_ids = torch.zeros(H, W, 3)

        H_index = torch.arange(H)
        W_index = torch.arange(W)
        if scale is not None:
            H_index = H_index / (H-1) * (scale-1)
            W_index = W_index / (W-1) * (scale-1)
        img_ids[..., 1] = img_ids[..., 1] + H_index[:, None]
        img_ids[..., 2] = img_ids[..., 2] + W_index[None, :]
        latent_image_id_height, latent_image_id_width, latent_image_id_channels = img_ids.shape
        img_ids = img_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        img_ids = img_ids.to(device=device, dtype=dtype)


        ids = torch.cat((txt_ids, img_ids), dim=0)

        image_rotary_emb = self.pos_embed(ids)
        return image_rotary_emb
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
   
        hidden_states = self.patch_embedder(hidden_states) # n c h w

        scale = self.scale

        N, C, H, W = hidden_states.shape
        image_rotary_emb = self.get_pos_embed(hidden_states, H, W, scale)

        hidden_states = rearrange(hidden_states, 'n c h w -> n (h w) c')

        latent_tkn = self.latent_tokens.repeat(hidden_states.shape[0], 1, 1)
        hidden_states = torch.cat([latent_tkn, hidden_states], dim=1)
        
        down_idx = 0
        for index_block, block in enumerate(self.transformer_blocks):
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
                    image_rotary_emb,
                    use_reentrant=False,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    image_rotary_emb=image_rotary_emb,
                )

            if index_block in self.downsample_idx:
                latent_tkns = hidden_states[:,:self.num_latent_tkns, :]
                img_tkns = rearrange(hidden_states[:,self.num_latent_tkns:, :], 'n (h w) l -> n h w l', h=H, w=W)
                img_tkns = self.down_layers[down_idx](img_tkns)
                hidden_states = torch.cat([latent_tkns, img_tkns], dim=1)

                down_idx += 1
                H = H//2
                W = W//2

                if scale is not None:
                    scale = scale // 2

                image_rotary_emb = self.get_pos_embed(hidden_states, H, W, scale)

        hidden_states = hidden_states[:,:self.num_latent_tkns, :]
        hidden_states = self.norm_out(hidden_states)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return output

        return Transformer2DModelOutput(sample=output)

class AE_Decoder(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 768,
        num_layers: int = 12,
        attention_head_dim: int = 64,
        num_attention_heads: int = 16,
        upsample_idx: List[int] = [],
        axes_dims_rope: List[int] = [8, 28, 28],
        repa_dim: int=1024,
        repa_index: int=16,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.input_layer = nn.Sequential(
            LlamaRMSNorm(in_channels, 1e-6),
            nn.Linear(in_channels, self.inner_dim)
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads = num_attention_heads , 
                )
                for _ in range(num_layers)
            ]
        )

        self.upsample_idx = upsample_idx
        if len(upsample_idx)>=1:
            self.up_layers = nn.ModuleList([DCUpBlock2d(dim=self.inner_dim) for _ in range(len(upsample_idx))])

        self.norm_out = LlamaRMSNorm(self.inner_dim, 1e-6)

        self.mask_tokens = nn.Parameter(torch.zeros(1, 1, self.inner_dim))
        self.mask_tokens.data.normal_(0.0, 0.02)

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        if repa_dim > 0:
            self.repa_proj = nn.Sequential(
                nn.Linear(self.inner_dim, self.inner_dim),
                nn.LayerNorm(self.inner_dim),
                nn.ReLU(),
                nn.Linear(self.inner_dim, repa_dim),
            )
        self.repa_index = repa_index

        self.gradient_checkpointing = False
        self.scale = None

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def get_pos_embed(self, hidden_states, H, W, scale=None):
        dtype = self.input_layer[1].weight.dtype
        device = hidden_states.device

        num_latent_tkns = hidden_states.shape[1]
        txt_ids = torch.zeros(num_latent_tkns, 3)
        txt_ids[:, 0] = txt_ids[:, 0] + torch.arange(num_latent_tkns)
        txt_ids = txt_ids.to(device=device, dtype=dtype)

        img_ids = torch.zeros(H, W, 3)

        H_index = torch.arange(H)
        W_index = torch.arange(W)
        if scale is not None:
            H_index = H_index / (H-1) * (scale-1)
            W_index = W_index / (W-1) * (scale-1)
        img_ids[..., 1] = img_ids[..., 1] + H_index[:, None]
        img_ids[..., 2] = img_ids[..., 2] + W_index[None, :]
        latent_image_id_height, latent_image_id_width, latent_image_id_channels = img_ids.shape
        img_ids = img_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        img_ids = img_ids.to(device=device, dtype=dtype)


        ids = torch.cat((txt_ids, img_ids), dim=0)

        image_rotary_emb = self.pos_embed(ids)
        return image_rotary_emb

    def forward(
        self,
        latent_tokens: torch.Tensor,
        H: int,
        W: int,
        return_dict: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

        latent_tokens = self.input_layer(latent_tokens)
        num_latent_tkns = latent_tokens.shape[1]
        
        if len(self.upsample_idx) >= 1:
            H = H // (2 * len(self.upsample_idx))
            W = W // (2 * len(self.upsample_idx))
            if self.scale is not None:
                scale = self.scale // (2 * len(self.upsample_idx))
            else:
                scale = None

        num_img_tokens = H * W
        hidden_states = self.mask_tokens.repeat(latent_tokens.shape[0], num_img_tokens, 1)
        image_rotary_emb = self.get_pos_embed(latent_tokens, H, W, scale)

        hidden_states = torch.cat([latent_tokens, hidden_states], dim=1)

        up_idx = 0
        return_repa = False
        output_dict = {}
        for index_block, block in enumerate(self.transformer_blocks):
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
                    image_rotary_emb,
                    use_reentrant=False,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    image_rotary_emb=image_rotary_emb,
                )

            if index_block == self.repa_index and self.training:
                repa_feature = self.repa_proj(hidden_states[:,num_latent_tkns:, :])
                output_dict['repa_feature'] = repa_feature
            
            if index_block in self.upsample_idx:
                latent_tkns = hidden_states[:,:num_latent_tkns, :]
                img_tkns = rearrange(hidden_states[:,num_latent_tkns:, :], 'n (h w) l -> n h w l', h=H, w=W)
                img_tkns = self.up_layers[up_idx](img_tkns)
                # img_tkns = rearrange(img_tkns, 'n h w l -> n (h w) l')
                hidden_states = torch.cat([latent_tkns, img_tkns], dim=1)

                up_idx += 1
                H = H*2
                W = W*2
                if scale is not None:
                    scale = scale * 2
                image_rotary_emb = self.get_pos_embed(latent_tkns, H, W, scale)
            # print(hidden_states.shape)

        hidden_states = hidden_states[:,num_latent_tkns:, :]
        hidden_states = self.norm_out(hidden_states)
        output_dict['hidden_states'] = hidden_states
        return output_dict



class BAE_Model(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.configs = config
        self.encoder = AE_Encoder(in_channels=config.in_channels,
                                    num_layers=config.encoder_layers,
                                    attention_head_dim=config.encoder_head_dim,
                                    num_attention_heads=config.encoder_num_heads,
                                    patch_size=config.patch_size,
                                    num_latent_tkns=config.num_latent_tkns,
                                    downsample_idx=config.downsample_idx,
                                    axes_dims_rope=config.axes_dims_rope,
        )

        latent_dim = config.encoder_head_dim * config.encoder_num_heads

        use_repa = config.get('use_repa', False)

        self.decoder = AE_Decoder(
                                    in_channels=latent_dim,
                                    num_layers=config.decoder_layers,
                                    attention_head_dim=config.decoder_head_dim,
                                    num_attention_heads=config.decoder_num_heads,
                                    upsample_idx=config.upsample_idx,
                                    axes_dims_rope=config.axes_dims_rope,
                                    repa_dim=config.get('repa_dim', None) if use_repa else -1,
                                    repa_index=config.get('repa_index', None) if use_repa else -1,
        )
        
        self.to_pixel = nn.Linear(config.decoder_head_dim * config.decoder_num_heads, config.patch_size*config.patch_size*config.in_channels)
        self.quantizer = BinaryQuantizer(config.codebook_size, latent_dim, config.num_latent_tkns, train_temp=config.get('train_temp', 0.1))
        self.in_channels = config.in_channels
    
    def set_scale(self, scale):
        scale = scale /  self.configs.patch_size
        self.encoder.scale = scale
        self.decoder.scale = scale

    def enable_gradient_checkpointing(self):
        self.encoder.gradient_checkpointing = True
        self.decoder.gradient_checkpointing = True

    def encode(
        self,
        hidden_states: torch.Tensor,
    ):
        hidden_states = self.encoder(hidden_states)
        hidden_states, code_loss, info = self.quantizer(hidden_states)

        return hidden_states, code_loss, info
    
    def decode(
        self,
        hidden_states,
        H,
        W,
    ):
        hidden_states = torch.einsum("b l n, n d -> b l d", hidden_states, self.quantizer.embed)
        hidden_states = self.decoder(hidden_states, H, W).pop('hidden_states')
        N = hidden_states.shape[0]

        hidden_states = self.to_pixel(hidden_states)
        hidden_states = hidden_states.view(N, H, W, self.configs.patch_size, self.configs.patch_size, self.in_channels)
        output = rearrange(hidden_states, 'n h w a b c-> n c (h a) (w b)')
        return output
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        H: int = None,
        W: int = None,
    ):

        N, C, h, w = hidden_states.shape
        if H is None:
            H = h
            W = w

        H = H // self.configs.patch_size
        W = W // self.configs.patch_size

        hidden_states = self.encoder(hidden_states)

        hidden_states, _, info = self.quantizer(hidden_states)

        output_dec = self.decoder(hidden_states, H, W)
        hidden_states = output_dec.pop('hidden_states')


        hidden_states = self.to_pixel(hidden_states)
        hidden_states = hidden_states.view(N, H, W, self.configs.patch_size, self.configs.patch_size, self.in_channels)
        output = rearrange(hidden_states, 'n h w a b c-> n c (h a) (w b)')
        
        output_dec['hidden_states'] = output
        return output_dec

if __name__ == '__main__':
    model = AE_Decoder()
    print(model)