# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class BinaryQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, num_latent_tokens, train_temp=1.0, use_tanh=False):
        super().__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        act = nn.Sigmoid
        if use_tanh:
            act = nn.Tanh
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, codebook_size),  # projects last encoder layer to quantized logits
            # act(),
            )
        # self.embed = nn.Embedding(codebook_size, emb_dim)
        # self.embed = nn.Parameter(torch.zeros(num_latent_tokens, codebook_size, emb_dim))
        self.embed = nn.Parameter(torch.zeros(codebook_size, emb_dim))
        self.embed.data.normal_(0.0, 0.02)
        self.use_tanh = use_tanh
        self.train_temp = train_temp

    def quantizer(self, x, deterministic=False):
        if deterministic or (not self.training):
            x = ((torch.sigmoid(x) > 0.5) * 1.0).to(dtype=x.dtype)
            return x
        else:
            return torch.bernoulli(torch.sigmoid(x / self.train_temp))

    def forward(self, h, deterministic=False):

        z = self.proj(h)
        z_normed = torch.sigmoid(z)
        # code_book_loss = F.binary_cross_entropy_with_logits(z, (torch.sigmoid(z.detach())>0.5)*1.0)
        code_book_loss = (z_normed * (1 - z_normed)).mean()

        z_b = self.quantizer(z, deterministic=deterministic)

        z_flow = z_b.detach() + z_normed - z_normed.detach()

        z_q = torch.einsum("b l n, n d -> b l d", z_flow, self.embed)
        # z_q = torch.einsum("b l n, l n d -> b l d", z_flow, self.embed)

        return z_q,  code_book_loss, {
            "binary_code": z_b.detach()
        }
