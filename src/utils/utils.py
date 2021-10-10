import os
import math
import json
import shutil
import torch
import numpy as np
from dotmap import DotMap


def load_json(f_path):
    with open(f_path, 'r') as f:
        return json.load(f)


def save_json(obj, f_path):
    with open(f_path, 'w') as f:
        json.dump(obj, f, ensure_ascii=False)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def kl_normal_normal(mean_p, logvar_p, mean_q, logvar_q, eps=1e-7):
    logvar_ratio = logvar_p - logvar_q
    var_ratio = torch.exp(logvar_ratio) + eps
    var_q = torch.exp(logvar_q) + eps
    t1 = (mean_p - mean_q).pow(2) / var_q
    kl_div = 0.5 * (var_ratio + t1 - 1 - logvar_ratio)
    return torch.sum(kl_div, dim=1)


def kl_standard_normal(mu, logvar):
    kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_div = torch.sum(kl_div, dim=1)
    return kl_div

