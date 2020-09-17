import torch
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
from collections import Counter
from torch.autograd import Variable


def prob_margins(scores, labels, softmax=True):
    """
    return prob margins
    :param scores: (N, |V|)
    :param labels: (N,)
    :param softmax: Do softmax if True.
    :return: prob margins, (N,)
    """
    if softmax:
        probs = F.softmax(scores, -1)
    else:
        probs = scores

    labels_index = torch.unsqueeze(labels, dim=1)  # (N, 1)
    label_probs = torch.gather(probs, 1, labels_index)  # (N, 1)
    label_masks = torch.BoolTensor(probs.size()).fill_(0)
    if scores.is_cuda:
        label_masks = label_masks.cuda()
    label_masks = label_masks.scatter_(1, labels_index, 1)
    probs_wo_label = probs.masked_fill(mask=label_masks, value=0)
    max_vals, _ = probs_wo_label.max(dim=-1, keepdim=True)
    margins = (label_probs - max_vals).squeeze(dim=1)
    return margins


def rm_padding_scores(scores, labels, padding_idx=0):
    """
    remove padding scores
    :param scores: (bsz, seqlen, |V|)
    :param labels: (bsz, seqlen)
    :param padding_idx: int
    :return: tensor of shape (N, |V|), tensor of shape (N,)
    """

    padding_stops = (labels != padding_idx).sum(dim=1)
    scores_no_padding = scores[0][0:padding_stops[0], :]
    labels_no_padding = labels[0][0:padding_stops[0]]

    for b_idx in range(1, len(scores)):
        b_score = scores[b_idx]
        scores_no_padding = torch.cat(
            [scores_no_padding, b_score[0:padding_stops[b_idx], :]], dim=0
        )
        labels_no_padding = torch.cat(
            [labels_no_padding, (labels[b_idx][0:padding_stops[b_idx]])], dim=0
        )
    assert len(scores_no_padding) == len(labels_no_padding)
    return scores_no_padding, labels_no_padding


def desc_of_tensor(t):
    """
    return describe numbers (7) of a tensor t.
    :param t:
    :return: tensor of shape (7,)
    """
    use_cuda = t.is_cuda
    assert len(t.size()) == 1, '{} must be a 1-dim tensor!'.format(t)
    if use_cuda:
        t = t.cpu()
    # noinspection PyUnresolvedReferences
    desc = torch.FloatTensor(pd.Series(t).describe().array[1:])
    if use_cuda:
        desc = desc.cuda()
    return desc


def create_batch_from_file(input_file, batch_size=128):
    batch_arr = []
    assert batch_size >= 1
    assert os.path.exists(input_file), "{} does not exist!".format(input_file)
    with open(input_file) as f:
        data = [line.strip() for line in f.readlines()]

    for i in range(0, len(data), batch_size):
        stop = i + batch_size
        stop = min(len(data), stop)
        batch_arr.append([data[j] for j in range(i, stop)])
    return batch_arr


def texts_to_bow(texts, vocab_size, special_token_idxs=None):
    bows = []
    if type(texts) is torch.Tensor:
        texts = texts.tolist()
    for sentence in texts:
        bow = Counter(sentence)
        # Remove special tokens
        if special_token_idxs is not None:
            for idx in special_token_idxs:
                bow[idx] = 0

        x = np.zeros(vocab_size, dtype=np.int64)
        x[list(bow.keys())] = list(bow.values())
        bows.append(torch.FloatTensor(x).unsqueeze(dim=0))
    bows = Variable(torch.cat(bows, dim=0))
    return bows


def override_opt(opt, OVERRIDE):
    opt['override'] = opt['override'] if 'override' in opt else {}
    # Add arguments of OVERRIDE into opt or opt['override']
    for k, v in OVERRIDE.items():
        if k not in opt:
            print("[ Add {} with value {} to opt ]".format(k, v))
            opt[k] = v
        else:
            # Add {k: opt[k]} to opt['override']
            opt['override'][k] = opt[k]
            print("[ Add {} with value {} to opt['override'] ]".format(k, opt[k]))
    return opt
