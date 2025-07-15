# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch, os

from models.ar_model import Instella_AR_Model, BinaryAR

import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from bae.ae_model import BAE_Model
import gc
from mmengine.config import Config
import argparse
from safetensors.torch import load_file as safe_load_file
from huggingface_hub import snapshot_download

def parse_args():
    parser = argparse.ArgumentParser(description="Testing configuration for Instella diffusion model.")

    parser.add_argument("--ckpt_path", type=str, default='./checkpoints', help='Path to the diffusion model ckpt')
    parser.add_argument("--num_tkn", type=int, default=128, help='Number of image tokens for the BAE tokenizer')
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 768, 1024], help='output image size')
    parser.add_argument("--codesize", type=int, default=128, help='Codebook size of the BAE tokenizer')
    parser.add_argument("--cfg_scale", type=float, default=7.5, help='Scale of classifier-free guidance')
    parser.add_argument("--temp", type=float, default=1.0, help='Temperature for sampling')
    parser.add_argument("--rho", type=float, default=1.0, help='Rho for sampling')
    parser.add_argument("--num_steps", type=int, default=20, help='Number of inference steps')
    parser.add_argument("--sampling_protocal", type=str, default='protocal_1', choices=['protocal_1', 'protocal_2'], help='Sampling protocal')
    parser.add_argument("--config", type=str, default='configs/diff_config.py', help='Path to the config file')
    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    # Model download
    local_path = snapshot_download(
            repo_id="amd/Instella-T2I",
            local_dir=args.ckpt_path, 
        )

    print("Model downloaded to:", local_path)

    model_config = Config.fromfile(args.config)
    bae_config = Config.fromfile(model_config.bae_config)
    bae_ckpt = model_config.bae_ckpt

    weight_dtype = torch.bfloat16

    # ======================================================
    # Build models
    # ======================================================
    print('Build diffusion model and load weight')
    olmo_path = 'amd/AMD-OLMo-1B'
    llm = AutoModelForCausalLM.from_pretrained(olmo_path, attn_implementation="flash_attention_2", torch_dtype=weight_dtype).to("cuda") # remove .to("cuda") to load on cpu
    tokenizer = AutoTokenizer.from_pretrained(olmo_path, model_max_length=128, padding_side='left')



    model = Instella_AR_Model(
                                in_channels = bae_config.codebook_size,
                                num_layers = llm.config.num_hidden_layers,
                                attention_head_dim = model_config.get('attention_head_dim', 128),
                                num_attention_heads = model_config.get('num_attention_heads', 16),
                                num_img_tkns = model_config.num_tkns,
                                text_cond_dim = llm.config.hidden_size,
                                )

    model.eval()



    ckpt = torch.load(f'{args.ckpt_path}/ar.pt', map_location='cpu')['module']
    model.load_state_dict(ckpt)

    bae = BAE_Model(bae_config)

    print('Loading BAE model weights')
    bae_state_dict = safe_load_file(bae_ckpt)
    bae.load_state_dict(bae_state_dict, strict=True)
    del bae_state_dict
    gc.collect()
    torch.cuda.empty_cache()


    bae.to('cuda', dtype=weight_dtype)
    bae.eval()
    if model_config.get('bae_scale', None) is not None:
        bae.set_scale(model_config.bae_scale)

    bae.requires_grad_(False)

    model = model.to('cuda', dtype=weight_dtype)

    num_sampling_steps = args.num_steps
    img_size = args.image_size
    
    guidance_scale = args.cfg_scale
    temp = args.temp

    binary_ar = BinaryAR(model_config.num_tkns)

    os.makedirs('results', exist_ok=True)

    while True:
       
        text = str(input('Enter your prompt: '))
        if text == 'exit':
            exit()

        try:
            with torch.no_grad():
                ts = time.time()
                z = binary_ar.sample(model, tokenizer, llm, [text], guidance_scale=guidance_scale, temp=temp)
                z = z.to(weight_dtype)

                samples = bae.decode(z, img_size//16, img_size//16)
                te = time.time()

                samples = samples[0].float()

                samples = torch.clamp(samples, -1.0, 1.0)
                samples = (samples + 1) / 2

                samples = samples.permute(1, 2, 0).mul_(255).cpu().numpy()
                image = Image.fromarray(samples.astype(np.uint8))
                name = text.split(' ')[:5]
                name = '_'.join(name)
                image.save(f'results/{img_size}_{num_sampling_steps}_{guidance_scale}_{temp}_{name}.jpg')
                sp = te - ts
            print(f'Generation finished in {sp:.2f}s')
        except Exception as e:
            print(f'Skipping due to error: {e}')
            continue


if __name__ == "__main__":
    main()
