from typing import List, Sequence
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
from addict import Dict

from fcore.utils.structure import GenericData

def sharpen(x: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    sharpened_x = x ** (1 / temperature) 
    return sharpened_x / sharpened_x.sum(dim=1, keepdim=True) 

def entropy(x):
    p = torch.softmax(x, dim=1)
    entropy = -(p * p.log()).sum(dim=1)
    return entropy

def plot_twin(_y1, _y2, _ylabel1, _ylabel2, _xlabel):
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel(_xlabel)
    ax1.set_ylabel(_ylabel1, color=color)
    ax1.plot(_y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel(_ylabel2, color=color)
    ax2.plot(_y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

def time_test(func):
    def wrapper(*args):
        st = time.time()
        out = func(*args)
        print(args[0].__class__.__name__, "{:.8f}".format(time.time() - st))
        return out

    return wrapper


def attach_batch_idx(tensor_list):
    tensors = []
    for i, tensor in enumerate(tensor_list):
        batch_idx = torch.ones((tensor.shape[0], 1), dtype=tensor.dtype, device=tensor.device) * i
        tensors.append(torch.cat((batch_idx, tensor), dim=1))
    return torch.cat(tensors, dim=0)


def print_tensor(x, device="cuda:0"):
    if isinstance(x, torch.Tensor):
        if str(x.device) == device:
            print(x)
    else:
        print(x)

def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    # mask = torch.zeros(img_size, img_size)
    mask = torch.zeros(img_size)
    if random.random() > p:
        if mask[0].shape != 512 or mask[1].shape != 512:
            pad_row = 512 - mask.size(0)
            pad_col = 512 - mask.size(1)
            mask = F.pad(mask, (0, pad_col, 0, pad_row), value=0.)
        return mask

    size = np.random.uniform(size_min, size_max) * img_size[0] * img_size[1]
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size[0])
        y = np.random.randint(0, img_size[1])

        if x + cutmix_w <= img_size[0] and y + cutmix_h <= img_size[1]:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1
    if mask[0].shape != 512 or mask[1].shape != 512:
        pad_row = 512 - mask.size(0)
        pad_col = 512 - mask.size(1)
        mask = F.pad(mask, (0, pad_col, 0, pad_row), value=0.)

    return mask

def percent(cur_step, total_step):
    return 1 - (cur_step / total_step)

def slerp(low, high, val=0.5):
    omega = torch.arccos((low * high).sum(dim=1, keepdim=True))
    so = torch.sin(omega)
    return torch.sin((1.0 - val) * omega) / so * low + torch.sin(val * omega) / so * high

def split_ab(x):
    if len(x) <= 1:
        return x, x
    a = x[: int(len(x) / 2)]
    b = x[int(len(x) / 2) :]

    return a, b

def interleave_offsets(batch_size: int, num_unlabeled: int) -> List[int]:
    # TODO: scrutiny
    groups = [batch_size // (num_unlabeled + 1)] * (num_unlabeled + 1)
    for x in range(batch_size - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch_size
    return offsets


def interleave(xy: Sequence[GenericData], batch_size: int) -> List[GenericData]:
    # TODO: scrutiny
    num_unlabeled = len(xy) - 1
    offsets = interleave_offsets(batch_size, num_unlabeled)
    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(num_unlabeled + 1)] for v in xy]
    for i in range(1, num_unlabeled + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]

    outs = []
    for v in xy:
        if torch.is_tensor(v[0]):
            v = torch.cat(v, dim=0)
        else:
            v = [item for subv in v for item in subv]
        outs.append(v)
    return outs


def one_hot(y, num_classes):
    y_tensor = y.unsqueeze(1)
    zeros = torch.zeros([y.shape[0], num_classes] + list(y.shape[1:]), dtype=y.dtype, device=y.device)

    return zeros.scatter(1, y_tensor, 1)

def topK(logits, ground_truth, x=None):
    # logits: B * C * N * H
    # ground_truth: B * N * H
    # x: int
    rank_counts = torch.zeros(logits.shape[1], dtype=torch.int32)
    rank_confidence_0 = torch.zeros(5, dtype=torch.int32)
    rank_confidence_1 = torch.zeros(5, dtype=torch.int32)
    rank_confidence_2 = torch.zeros(5, dtype=torch.int32)
    rank_confidence_other = torch.zeros(5, dtype=torch.int32)
    rank_confidence = Dict()
    pseudo = torch.argmax(logits, dim=1)
    mask = ~(pseudo == ground_truth)
    for batch in range(logits.shape[0]):
        for i in range(logits.shape[-1]):
            for j in range(logits.shape[-1]):
                if ground_truth[batch, i, j] == 255:
                    continue
                label = ground_truth[batch, i, j]
                label_logit = logits[batch, label, i, j]
                rank = (logits[batch, :, i, j] > label_logit).sum().item()
                rank_counts[rank] += 1
                # if mask[batch, i, j]:
                #     if label == 255:
                #         continue
                #     value = logits[batch, label, i, j]
                #     rank_confidence[int(value * 5)] += 1
                value = torch.where(logits[batch, label, i, j] >= 1, 0.9999, logits[batch, label, i, j])
                if rank == 0:
                    rank_confidence_0[int(value * 5)] += 1
                elif rank == 1:
                    rank_confidence_1[int(value * 5)] += 1
                elif rank == 2:
                    rank_confidence_2[int(value * 5)] += 1
                else:
                    rank_confidence_other[int(value * 5)] += 1
    rank_confidence.rank1 = rank_confidence_0
    rank_confidence.rank2 = rank_confidence_1
    rank_confidence.rank3 = rank_confidence_2
    rank_confidence.rank4 = rank_confidence_other

    return rank_counts, rank_confidence