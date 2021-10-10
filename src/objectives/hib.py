import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from src.utils import utils

class HIB(object):

    def __init__(self, loc1, stdev1, loc2, stdev2, a, b, beta=1e-4, K=8, t=0.07, eps=1e-6):
        super().__init__()
        self.loc1 = loc1
        self.loc2 = loc2
        self.stdev1 = stdev1
        self.stdev2 = stdev2
        self.a, self.b = a, b
        self.beta = beta
        self.K = K
        self.eps, self.t = eps, t

    def batch_contrastive(self, loc1, stdev1, loc2, stdev2, a, b, match=True):
        # Build the distributions
        normal1 = Normal(loc1, stdev1)
        normal2 = Normal(loc2, stdev2)
        
        # Sample K from each distribution
        samples1 = []
        samples2 = []
        for i in range(self.K):
            samples1.append(normal1.rsample())
            samples2.append(normal2.rsample())
        
        # Get the K^2 distances per sample
        dists = []
        for i in range(self.K):
            for j in range(self.K):
                dist = torch.norm(samples1[i] - samples2[j], dim=1)
                dists.append(dist)
        dists = torch.stack(dists, dim=1)
        dists = F.sigmoid(-a * dists + b)
        prob = dists.mean(1)
        
        # Final contrastive loss
        if match:
            return F.binary_cross_entropy(prob, torch.ones_like(prob))
        else:
            return F.binary_cross_entropy(prob, torch.zeros_like(prob))

    def divergence(self, loc, stdev): # Do I use variance or Stdev?
        log_var = torch.log(stdev ** 2)
        return utils.kl_standard_normal(loc, log_var)

    def get_loss(self, split=False):
        # loss = expected contrastive loss - B * [KL (p(z1|x1) || r(z)) + KL(p(z2|x2)||r(z))]
        # where r(z) is the prior (i.e., the standard normal) 
        pos_contrastive = self.batch_contrastive(self.loc1, self.stdev1, self.loc2, 
            self.stdev2, self.a, self.b, match=True)

        loc2_neg = torch.roll(self.loc2, 1, 0)
        stdev2_neg = torch.roll(self.stdev2, 1, 0)
        neg_contrastive = self.batch_contrastive(self.loc1, self.stdev1, loc2_neg, 
            stdev2_neg, self.a, self.b, match=False)

        contrastive = pos_contrastive + neg_contrastive

        divergence1 = self.divergence(self.loc1, self.stdev1).mean()
        divergence2 = self.divergence(self.loc2, self.stdev2).mean()

        return contrastive + self.beta * (divergence1 + divergence2)
