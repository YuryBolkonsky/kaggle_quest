import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ListMLE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        shuffled_indices = torch.argsort(torch.rand_like(logits), 0)
        shuffled_labels = torch.gather(labels, 0, shuffled_indices)
        shuffled_logits = torch.gather(logits, 0, shuffled_indices)

        sorted_indices = torch.argsort(shuffled_labels, 0)
        sorted_labels, sorted_logits = torch.gather(
            shuffled_labels, 0, sorted_indices), torch.gather(shuffled_logits, 0, sorted_indices)
        raw_max = torch.max(sorted_logits, dim=0, keepdim=True)[0]
        sorted_logits = sorted_logits - raw_max

        sums = torch.cumsum(torch.exp(sorted_logits), 0)
        sums = torch.log(sums) - sorted_logits
        loss = sums.mean()
        return loss

class ListMLEQueue(nn.Module):
    def __init__(self, queue_size = 8):
        super().__init__()
        self.queue_size = queue_size
        self.logits_queue = []
        self.labels_queue = []

    def forward(self, logits, labels):
        if len(self.logits_queue):
            self.logits_queue[-1] = self.logits_queue[-1].detach()
            self.labels_queue[-1] = self.labels_queue[-1].detach()
        self.logits_queue.append(logits)
        self.labels_queue.append(labels)
        if len(self.logits_queue) > self.queue_size:
            self.logits_queue.pop(0)
            self.labels_queue.pop(0)
        logits = torch.cat(self.logits_queue,0)
        labels = torch.cat(self.labels_queue,0)

        shuffled_indices = torch.argsort(torch.rand_like(logits), 0)
        shuffled_labels = torch.gather(labels, 0, shuffled_indices)
        shuffled_logits = torch.gather(logits, 0, shuffled_indices)

        sorted_indices = torch.argsort(shuffled_labels, 0)
        sorted_labels, sorted_logits = torch.gather(
            shuffled_labels, 0, sorted_indices), torch.gather(shuffled_logits, 0, sorted_indices)
        raw_max = torch.max(sorted_logits, dim=0, keepdim=True)[0]
        sorted_logits = sorted_logits - raw_max

        sums = torch.cumsum(torch.exp(sorted_logits), 0)
        sums = torch.log(sums) - sorted_logits
        loss = sums.mean()
        return loss


class ListMLEDistributed(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args

    def forward(self, logits, labels):
        gather_logits = [torch.zeros_like(logits)
                         for _ in range(self.args.world_size)]
        gather_labels = [torch.zeros_like(labels)
                         for _ in range(self.args.world_size)]
        torch.distributed.all_gather(gather_logits, logits.detach())
        torch.distributed.all_gather(gather_labels, labels.detach())
        del gather_logits[self.args.local_rank], gather_labels[self.args.local_rank]
        logits = torch.cat(
            [logits, gather_logits], 0)
        labels = torch.cat(
            [labels, gather_labels], 0)

        shuffled_indices = torch.argsort(torch.rand_like(logits), 0)
        shuffled_labels = torch.gather(labels, 0, shuffled_indices)
        shuffled_logits = torch.gather(logits, 0, shuffled_indices)

        sorted_indices = torch.argsort(shuffled_labels, 0)
        sorted_labels, sorted_logits = torch.gather(
            shuffled_labels, 0, sorted_indices), torch.gather(shuffled_logits, 0, sorted_indices)
        raw_max = torch.max(sorted_logits, dim=0, keepdim=True)[0]
        sorted_logits = sorted_logits - raw_max

        sums = torch.cumsum(torch.exp(sorted_logits), 0)
        sums = torch.log(sums) - sorted_logits
        loss = sums.mean()
        return loss
