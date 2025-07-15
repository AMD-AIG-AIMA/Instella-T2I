# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.

import os, shutil, math, sys
sys.path.insert(0, 'packages')
import logging, gc, time
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torchvision.utils import make_grid
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed

from tqdm import tqdm
from pathlib import Path
from utils.config_utils import parse_configs
from utils.lr_scheduler import LinearWarmupLR
from utils.misc import (
    format_numel_str,
    get_model_numel,
    random_crop_arr
)
from utils.training_utils import logistic_normal_t_sample, imagenet_prompt_translate

import transformers
import diffusers
from diffusers.utils import  is_wandb_available
from transformers import AutoModelForCausalLM, AutoTokenizer
from mmengine.config import Config
from safetensors.torch import load_file as safe_load_file
import wandb

from models.diff_model import Instella_Binary_Diff, BinaryDiffusion
from bae.ae_model import BAE_Model


logger = get_logger(__name__)

class ProgressInfo:
    def __init__(self, global_step, train_loss=0.0):
        self.global_step = global_step
        self.train_loss = train_loss

def main():
    # ======================================================
    # Training confgurations
    # ======================================================
    #parse configs
    cfg = parse_configs(training=True)

    logging_dir = Path(cfg.outputs, cfg.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=cfg.outputs, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_config=accelerator_project_config,
        step_scheduler_with_optimizer=False,
    )
    print("Using device:", accelerator.device)


    if cfg.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.get("seed", None) is not None:
        set_seed(cfg.seed)
        np.random.seed(cfg.seed+dist.get_rank())
    else:
        np.random.seed(np.random.randint(1000)+dist.get_rank())

    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.outputs is not None:
            os.makedirs(cfg.outputs, exist_ok=True)

    # data type, BFloat16 is recommended for training
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    # ======================================================
    # Building training datasets
    # ======================================================
    logger.info("Building dataset...")
    # == build dataset ==

    logger.info(f'Batch size: {cfg.get("batch_size")}')
    # == build dummy training dataset loader based on Imagenet ==

    transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, 512)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
    dataset = ImageFolder(cfg.imagenet_data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    prompt_translate = imagenet_prompt_translate()

    # ======================================================
    # Building models (training model and facilitation models)
    # ======================================================
    logger.info("Building models...")
    # == building text feature extractor based on pretrained LLM ==
    llm = AutoModelForCausalLM.from_pretrained('amd/AMD-OLMo-1B', attn_implementation="flash_attention_2", torch_dtype=weight_dtype).to("cuda") # remove .to("cuda") to load on cpu
    tokenizer = AutoTokenizer.from_pretrained('amd/AMD-OLMo-1B', model_max_length=cfg.get('text_max_length', 128), padding_side='left')
    
    # == building BAE for online image tokenization ==
    bae_config = cfg.bae_config
    bae_ckpt = cfg.bae_ckpt
    
    bae_config = Config.fromfile(bae_config)
    cfg.num_tkns = bae_config.num_latent_tkns
    bae = BAE_Model(bae_config)

    if cfg.get('bae_scale', None) is not None:
        bae.set_scale(cfg.bae_scale)

    logger.info('Loading BAE model weights')
    bae_state_dict = safe_load_file(bae_ckpt)
    bae.load_state_dict(bae_state_dict, strict=True)
    del bae_state_dict
    gc.collect()
    torch.cuda.empty_cache()

    bae.to(accelerator.device, dtype=weight_dtype)
    bae.eval()
    bae.requires_grad_(False)

    # == building training model (Instella_Binary_Diff) ==
    model = Instella_Binary_Diff(
                                in_channels = bae_config.codebook_size,
                                num_layers = llm.config.num_hidden_layers,
                                attention_head_dim = 128,
                                num_attention_heads = 16,
                                num_img_tkns = cfg.num_tkns,
                                text_cond_dim = llm.config.hidden_size,
                                )

    model.train()
    # model.gradient_checkpointing_enable()
    model.gradient_checkpointing = cfg.get("grad_checkpoint", True)

    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        "[Diffusion] Trainable model params: %s, Total model params: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel),
    )
    # == Model initialization. Note: model paramters only; this is not identical to training resume ==
    if cfg.get("init_from_path", None) is not None:
        ckpt = torch.load(cfg.init_from_path, map_location='cpu')['module']
        model.load_state_dict(ckpt)
        del ckpt
        gc.collect()
        torch.cuda.empty_cache()

    binary_diffusion = BinaryDiffusion(cfg.num_tkns, bae_config.codebook_size, weight_dtype)

    # ======================================================
    # Building optimizers and learning rate schedulers
    # ======================================================
    if cfg.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    lr = lr=cfg.get("lr", 1e-4)
    optimizer = optimizer_class(
        [{'params': model.parameters(), 'lr': lr}],
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("adam_eps", 1e-8),
    )
    warmup_steps = cfg.get("warmup_steps", None)

    if warmup_steps is None:
        lr_scheduler = None
    else:
        lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=cfg.get("warmup_steps"))

    # =======================================================
    # distributed training preparation with accelerate
    # =======================================================
    logger.info("Preparing for distributed training...")

    logger.info(f'before accelerator.prepare')
    model, optimizer, lr_scheduler, dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, dataloader
    )
    logger.info(f'after accelerator.prepare')

    
    exp_name = cfg.outputs.split('/')[-1]
    if accelerator.is_main_process:
        init_kwargs = {"wandb": {"name": exp_name, "dir": cfg.outputs}}
        accelerator.init_trackers('LLM_Diff', config=vars(cfg), init_kwargs=init_kwargs)

    train_batch_size = cfg.batch_size

    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")
    logger.info(f"  Total training parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9} B")
    global_step = 0

    # == load in the weights and states from a previous save ==
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            path = os.path.basename(cfg.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(cfg.outputs)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            cfg.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(cfg.outputs, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            
    else:
        initial_global_step = 0
    

    progress_bar = tqdm(
        range(0, cfg.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    progress_info = ProgressInfo(global_step, train_loss=0.0)

    def sync_gradients_info(loss, acc):
        progress_bar.update(1)
        progress_info.global_step += 1
        end_time = time.time()
        one_step_duration = end_time - start_time
        accelerator.log({"train_loss": progress_info.train_loss, "Binary Acc": acc}, step=progress_info.global_step)

        progress_info.train_loss = 0.0

        # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
        if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
            if progress_info.global_step % cfg.checkpointing_steps == 0:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if accelerator.is_main_process and cfg.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(cfg.outputs)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= cfg.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - cfg.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(cfg.outputs, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(cfg.outputs, f"checkpoint-{progress_info.global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")
        
        if accelerator.is_main_process and progress_info.global_step % cfg.vis_steps == 0:
            model.eval()
            with torch.no_grad():
                prompts = [
                    'Medium shot, Adorable creature with big reflective eyes, moody lighting, best quality, full body portrait, real picture, intricate details, depth of field, in a forest, fujifilm xt3, outdoors, bright day, beautiful lighting, raw photo, 8k uhd, film grain, unreal engine 5, ray tracing',
                    'Pirate ship sailing into a bioluminescence sea with a galaxy in the sky), epic, 4k, ultra,',
                    'Cute aesthetic, a (tiny cute translucent polycarbonate robot) with an LED screen face, emoticon, stunning unreal engine render, intricate details',
                    'Portrait of an old sea captain, male, detailed face, fantasy, highly detailed, cinematic, art painting by greg rutkowski.',
                ]
                
                z = binary_diffusion.sample(model, tokenizer, llm, prompts, num_sampling_steps=20, guidance_scale=7.5, temp=1.0, rho=1.0)

                z = z.to(weight_dtype)
                samples = bae.decode(z, cfg.resolution//16, cfg.resolution//16)
                samples = samples.float()
                samples = torch.clamp(samples, -1.0, 1.0)
            samples = make_grid((samples + 1) / 2, nrow=len(prompts), padding=0, pad_value=1.0)
            image = samples.permute(1, 2, 0).mul_(255).cpu().numpy()
            image = Image.fromarray(image.astype(np.uint8))
            accelerator.trackers[0].log({"gen_images": [wandb.Image(image)]}, step=progress_info.global_step)
            model.train()

        logs = {"step_loss": loss.detach().item(), "step_acc": acc, "lr": optimizer.param_groups[0]['lr']}
        progress_bar.set_postfix(**logs)

    # =======================================================
    # training loop
    # =======================================================
    while True:
        with accelerator.accumulate(model):
            for step, batch in enumerate(dataloader):

                global start_time
                start_time = time.time()

                x = batch[0].to(accelerator.device, dtype=weight_dtype)
                y = batch[1]

                y = prompt_translate(y, uncond_prob=cfg.get('uncond_prob', 0.1))

                # == visual and text encoding ==
                with torch.no_grad():
                    _, _, info  = bae.encode(x)
                    latents = info['binary_code']

                text_inputs = tokenizer(y, truncation=True, padding=True, max_length=cfg.get('text_max_length', 128), return_tensors='pt', return_token_type_ids=False).to("cuda")
                llm_features = llm(**text_inputs, output_hidden_states=True).hidden_states[1:]

                # == sampling timesteps ==
                if cfg.get('non_uniform_t', False):
                    t = logistic_normal_t_sample(latents).to(accelerator.device, dtype=weight_dtype)
                else:
                    t = torch.rand(latents.shape[0]).to(accelerator.device, dtype=weight_dtype)

                timepoints = t.view(-1, 1, 1)

                # == sampling latents ==
                x_t = binary_diffusion.add_noise(latents, timepoints)

                # == forward pass ==
                out = model(
                    hidden_states=x_t, 
                    llm_features=llm_features,
                    text_mask=text_inputs['attention_mask'],
                    timestep=t,
                )

                # == loss computation ==
                if cfg.get('prediction_target', 'epsilon') == 'x':
                    x_0_hat_logits = out
                elif cfg.get('prediction_target', 'epsilon') == 'epsilon':
                    x_0_hat_logits = x_t * ( - out) + (1 - x_t) * out
                else:
                    raise NotImplementedError

                loss = F.binary_cross_entropy_with_logits(x_0_hat_logits, latents)

                with torch.no_grad():
                    logits = (x_0_hat_logits > 0)*1.0
                    acc = ((logits == latents).sum() / latents.numel()).item() * 100
                loss_value = loss.item()

                if not math.isfinite(loss_value):
                    logger.info("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                # == logging ==
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                progress_info.train_loss += avg_loss.detach().item() / cfg.gradient_accumulation_steps
                if accelerator.sync_gradients:
                    sync_gradients_info(loss, acc)

                # update learning rate
                if lr_scheduler is not None:
                    lr_scheduler.step()

                if progress_info.global_step >= cfg.max_train_steps:
                    accelerator.wait_for_everyone()
                    accelerator.end_training()
                    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()



