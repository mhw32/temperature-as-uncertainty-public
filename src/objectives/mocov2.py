import math
import torch
import torch.nn.functional as F
from src.utils.utils import kl_normal_normal


class MoCoV2(object):

    def __init__(self, outputs_q, outputs_k, queue, t=0.07):
        super().__init__()
        self.outputs_q = F.normalize(outputs_q, dim=1)
        self.outputs_k = F.normalize(outputs_k, dim=1)
        self.queue = queue.clone().detach()
        self.t = t
        self.k = queue.size(0)
        self.device = outputs_q.device

    def get_loss(self):
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [self.outputs_q, self.outputs_k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [self.outputs_q, self.queue.t()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.t

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        loss = F.cross_entropy(logits.float(), labels.long())

        return loss


class TaU_MoCoV2(object):

    def __init__(self, outputs_q, temp_q, outputs_k, temp_k, queue, temp_queue, t=0.07):
        super().__init__()
        self.outputs_q = F.normalize(outputs_q, dim=1)
        self.outputs_k = F.normalize(outputs_k, dim=1)
        self.temp_q = torch.sigmoid(temp_q)
        self.temp_k = torch.sigmoid(temp_k)
        self.queue = queue.clone().detach()
        self.temp_queue = temp_queue.clone().detach()
        self.t = t
        self.k = queue.size(0)
        self.device = outputs_q.device

    def get_loss(self):
        l_pos = torch.sum(self.outputs_q * self.outputs_k, dim=-1) - 1
        l_pos = l_pos * (self.temp_q).squeeze(-1)
        l_pos = l_pos.unsqueeze(-1)
        
        cov = torch.mm(self.outputs_q, self.queue.t()) - 1
        var = self.temp_q.repeat(1, self.temp_queue.shape[0])
        l_neg = cov * var

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.t

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        loss = F.cross_entropy(logits.float(), labels.long())

        return loss
