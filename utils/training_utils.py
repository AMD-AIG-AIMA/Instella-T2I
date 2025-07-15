# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.

from torch.distributions import LogisticNormal
import torch
import json
import numpy as np

def logistic_normal_t_sample(x):
    t_distribution = LogisticNormal(torch.tensor([0.0]), torch.tensor([1.0]))
    sample_t = lambda x: t_distribution.sample((x.shape[0],))[:, 0].to(x.device)
    return sample_t(x)


class imagenet_prompt_translate(object):
    def __init__(self):
        with open('utils/imagenet_label.json', 'r') as f:
            self.name_dict = json.load(f)
    def __call__(self, index, uncond_prob=0.1):
        prompt_temp = []
        for p in index:
            if np.random.random() > uncond_prob:
                name = np.random.choice(self.name_dict[str(p.item())])
                p = f"A photo of {name}"
                prompt_temp.append(p)
            else:
                prompt_temp.append('')
        prompt = prompt_temp
        return prompt