# ======================================================
# Example training config for Instella Diffusion Model
# ======================================================

grad_checkpoint = True

# Acceleration settings
plugin = "zero2"
wandb = True
gradient_accumulation_steps = 1
report_to = 'wandb'
mixed_precision = 'bf16'

# Training settings
grad_clip = 1.0
lr = 1e-4
adam_eps = 1e-8
warmup_steps = 1000
use_8bit_adam = False


max_train_steps = int(2e6)
checkpointing_steps = 2500
checkpoints_total_limit = 10
log_every = 100
vis_steps = 2

non_uniform_t = True


# Dataset settings
imagenet_data_path = '/mnt/m2m_nobackup/vision_datasets/ImageNet_zip'
num_workers = 4
batch_size = 256


# Model settings
num_tkns = 128
attention_head_dim = 128
num_attention_heads = 16

text_max_length = 128

prediction_target = 'epsilon'


logging_dir = 'logging_outputs'
resume_from_checkpoint = 'latest'

resolution=512

num_tkns = 128

bae_config = 'configs/bae_config.py'
bae_ckpt = 'bae/model.safetensors'
bae_scale = 1024