# Instella-T2I: Pushing the Limits of 1D Discrete Latent Space Image Generation

<div align="center">
  <a href="https://arxiv.org/abs/2506.21022"><img src="https://img.shields.io/static/v1?label=Instella-T2I&message=ArXiv&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/amd/Instella-T2I"><img src="https://img.shields.io/static/v1?label=Instella-T2I&message=HuggingFace&color=green"></a> &ensp;
  <a href="https://rocm.blogs.amd.com/artificial-intelligence/instella-t2i/README.html"><img src="https://img.shields.io/static/v1?label=Instella-T2I&message=Blog&color=yellow"></a> &ensp;
</div>

<p align="center" border-radius="10px">
  <img src="assets/teaser.jpg" width="90%" alt="teaser"/>
</p>

Instella-T2I v0.1 is the first text-to-image model in the AMD Instella model family, trained exclusively using AMD Instinct MI300X GPUs. By representing images in a 1D binary latent space, our tokenizer encodes a 1024x1024 image using just 128 discrete tokens. Compared to the 4096 tokens typically required by standard VQ-VAEs, our tokenizer achieves a 32x token reduction. Instella-T2I v0.1 leverages our Instella-family language model, AMD OLMo-1B, for text encoding. The same architecture also serves as the backbone for both our diffusion and autoregressive models. Thanks to the large VRAM of the AMD Instinct MI300X GPUs and the compact 1D binary latent space adopted in Instella-T2I v0.1, we can fit 4096 images into a single computation node with 8 AMD Instinct MI300X GPUs, achieving a training throughput of over 220 images per second on each GPU. Both the diffusion and auto-regressive text-to-image models can be trained within 200 MI300X GPU days. Training Instella-T2I from scratch on AMD Instinct MI300X GPUs demonstrates the platform’s capability and scalability for a broad range of AI workloads, including computationally intensive text-to-image diffusion models.

## Getting Started

First install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system. For AMD GPUs, you can aslo start with a [rocm/pytorch](https://hub.docker.com/r/rocm/pytorch/tags?name=pytorch) docker.

To install the recommended packages, run: 

```bash
git clone https://github.com/AMD-AIG-AIMA/Instella-T2I.git
cd Instella-T2I
# install Flash-Attention on MI300X
GPU_ARCH=gfx942 MAX_JOBS=$(nproc) pip install git+https://github.com/Dao-AILab/flash-attention.git -v
# install other dependencies
pip install -r requirements.txt
```

## ▶️ Running the Tests

Using provide `test_diff.py` and `test_ar.py` to run image generation in interactive mode for the diffusion and AR models.

The inference scripts will automatically download the checkpoints to path specified by `--ckpt_path`.

```bash
python test_diff.py --ckpt_path DESIRED_PATH_TO_MODELS
python test_ar.py --ckpt_path DESIRED_PATH_TO_MODELS
```

#### Specifying hyperparameters
To specify hyperparameters, run:

```bash
python test_diff.py \
    --ckpt_path DESIRED_PATH_TO_MODELS \
    --cfg_scale 9.0 \
    --temp 0.8 \
    --num_steps 50 \
```

## 📝 Data

The training of the image generation models adopts a two-stage recipe. 
In stage one, the model is pretrained using the [LAION-COCO](https://huggingface.co/datasets/laion/laion-coco) dataset. In stage two, the data is augmented with synthetic image–text pairs, with a raio of 3:1 between the LAION and the synthetic data. The synthetic data consists of data from [Dalle-1M](https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions) and images generated from public models.

### Synthesis data
The training also includes a small amout of synthesis data.

The synthesis data are generated using the prompts from [DiffusionDB](https://github.com/poloclub/diffusiondb).
We use the following open models for generating the synthesis data:

- [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
- [Stable Diffusion 3.5 Large Turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo)
- [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
- [PixArt-Sigma](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS)

All data are generated using the models' defauls inference settings.


## 📖 Citation

If you find this project helpful for your research, please consider citing us:

```
@article{instella-t2i,
  title={Instella-T2I: Pushing the Limits of 1D Discrete Latent Space Image Generation},
  author={Wang, Ze and Chen, Hao and Hu, Benran and Liu, Jiang and Sun, Ximeng and Wu, Jialian and Su, Yusheng and Yu, Xiaodong and Barsoum, Emad and Liu, Zicheng},
  journal={arXiv preprint arXiv:2506.21022},
  year={2025}

```
