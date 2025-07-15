# ======================================================
# Example testing config for Instella 1D tokenizer Model
# ======================================================

#### Model configs
in_channels=3
encoder_layers=12
encoder_head_dim=64
encoder_num_heads=12
patch_size=16
num_latent_tkns=256
axes_dims_rope=[8, 28, 28]
downsample_idx=[7]

decoder_layers=16
decoder_head_dim=64
decoder_num_heads=16
upsample_idx=[7]

num_latent_tkns=128
codebook_size=64