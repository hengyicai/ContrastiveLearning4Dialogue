import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LabelSmoothing(nn.Module):
    """Implement adaptive label smoothing."""

    def __init__(self, size, padding_idx):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.padding_idx = padding_idx
        self.size = size

    def forward(self, x, target):
        """
        Label smoothing NLL loss
        :param x: log-prob, in the shape of (N, vocab_size)
        :param target: prob, in the shape of (N, vocab_size)
        :return: loss
        """
        assert x.size(1) == self.size
        # true_dist = x.data.clone()
        # true_dist.fill_(self.smoothing / (self.size - 2))
        # true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        target[:, self.padding_idx] = 0
        # mask = torch.nonzero(target.data == self.padding_idx)
        # if mask.size(0) > 0:
        #     true_dist.index_fill_(0, mask.squeeze(), 0.0)
        # self.true_dist = true_dist
        batch_loss = self.criterion(x, Variable(target, requires_grad=False))
        sum_loss = torch.sum(batch_loss)
        batch_loss = torch.sum(batch_loss, 1)
        return sum_loss, batch_loss


class CrossEntropyLabelSmoothing(LabelSmoothing):
    def __init__(self, size, padding_idx):
        super().__init__(size, padding_idx)

    def forward(self, x, target):
        return super().forward(F.log_softmax(x, 1), target)
