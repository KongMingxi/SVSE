import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_bce_loss(output, label, use_gpu=True):
    weight = torch.zeros(label.size())
    num_total = label.shape[0]
    if use_gpu:
        weight = weight.cuda()
    for i in range(label.shape[1]):
        is_pos = label.data[:, i] == 1
        num_neg = label.data[:, i] == 0
        num_neg = num_neg.sum()
        num_pos = label.data[:, i] == 1
        num_pos = num_pos.sum()
        weight[:, i][is_pos] = torch.true_divide(num_neg, num_total)
        weight[:, i][~is_pos] = torch.true_divide(num_pos, num_total)
        output = output.float()
        label = label.float()
        weight = weight.float()

    return F.binary_cross_entropy_with_logits(output, label, weight)


def bce_loss(output, label):

    return F.binary_cross_entropy_with_logits(output, label)


def nll_loss(output, target):
    return F.nll_loss(output, target)
