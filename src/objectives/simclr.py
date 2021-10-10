import math
import numpy as np
import torch
import torch.nn.functional as F


class SimCLR(object):

    def __init__(self, outputs1, outputs2, t=0.07, eps=1e-6):
        super().__init__()
        self.outputs1 = F.normalize(outputs1, dim=1)
        self.outputs2 = F.normalize(outputs2, dim=1)
        self.eps, self.t = eps, t

    def get_loss(self, split=False):
        # out: [2 * batch_size, dim]
        out = torch.cat([self.outputs1, self.outputs2], dim=0)
        n_samples = out.size(0)

        # cov and sim: [2 * batch_size, 2 * batch_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / self.t)

        mask = ~torch.eye(n_samples, device=sim.device).bool()
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(self.outputs1 * self.outputs2, dim=-1) / self.t)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + self.eps)).mean()

        if not split:
            return loss
        else:
            # return overall loss, numerator, denominator
            return loss, pos, (neg + self.eps)


class TaU_SimCLR(object):
    
    def __init__(self, loc1, temp1, loc2, temp2, t=0.07, eps=1e-6, simclr_mask=False):
        super().__init__()
        self.loc1 = F.normalize(loc1, dim=1)
        self.loc2 = F.normalize(loc2, dim=1)
        self.temp1 = torch.sigmoid(temp1)
        self.temp2 = torch.sigmoid(temp2)
        self.eps, self.t = eps, t
        self.simclr_mask = simclr_mask

    def build_mask(self, mb, device, simclr=False): # Either building the SimCLR mask or the new mask
        if simclr:
            m = torch.eye(mb, device=device).bool()
        else:
            m = torch.eye(mb // 2, device=device).bool()
            m = torch.cat([m, m], dim=1)
            m = torch.cat([m, m], dim=0)
        return m

    def get_loss(self, split=False):
        # out: [2 * batch_size, dim]
        loc = torch.cat([self.loc1, self.loc2], dim=0)
        temp = torch.cat([self.temp1, self.temp2], dim=0)
        n_samples = loc.size(0)

        # cov and sim: [2 * batch_size, 2 * batch_size]
        # neg: [2 * batch_size]
        cov = torch.mm(loc, loc.t().contiguous())
        var = temp.repeat(1, n_samples)
        sim = torch.exp((cov * var) / self.t)

        # NOTE: this mask is a little different than SimCLR's mask
        mask = ~self.build_mask(n_samples, sim.device, simclr=self.simclr_mask)
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.sum(self.loc1 * self.loc2, dim=-1)
        pos = pos * (self.temp1 / self.t).squeeze(-1)
        pos = torch.exp(pos)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + self.eps)).mean()

        if not split:
            return loss
        else:
            # return overall loss, numerator, denominator
            return loss, pos, (neg + self.eps)
