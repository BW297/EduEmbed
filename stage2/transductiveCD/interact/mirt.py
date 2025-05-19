import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def irt2pl(theta, a, b, *, F=np):

    return 1 / (1 + F.exp(- (F.sum(F.multiply(a, theta), axis=-1) - b)))


class MIRT_IF(nn.Module):
    def __init__(self,latent_dim, device, dtype):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.dtype = dtype
        self.a_range = 1

    @staticmethod
    def irt2pl(theta, a, b, F=torch):
        return 1 / (1 + F.exp(-(F.sum(F.multiply(a, theta), dim=1,keepdim=True) + b))).squeeze()

    def compute(self, student_ts, diff_ts, disc_ts, knowledge_ts, q_mask):
        theta = torch.squeeze(student_ts, dim=-1)
        a = torch.squeeze(diff_ts, dim=-1)
        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)
        b = torch.squeeze(disc_ts, dim=-1)
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The a_range is too large.')
        return self.irf(theta, a, b)


    @classmethod
    def irf(cls, theta, a, b):
        return irt2pl(theta, a, b, F=torch)
    def transform(self, mastery, knowledge):
        return torch.sigmoid(mastery)
