"""adapted from:
https://github.com/juho-lee/set_transformer
"""


import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from PIL import Image
import numpy
import matplotlib.cm as cm
import cv2


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)

        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SA(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SA, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class AE(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(AE, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class PICASO(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PICASO, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)      #shared parameters
        self.dim = dim
        self.num_seeds = num_seeds

    def forward(self, X):
        S_prime = self.mab(self.S.repeat(X.size(0), 1, 1), X)
        S_prime2 = self.mab(S_prime, X)
        S_prime3 = self.mab(S_prime2, X)

        return S_prime3

class Gen_PICASO(nn.Module):
    def __init__(self,  dim_in, dim, num_heads, num_seeds, ln=False, Vis=False):
        super(Gen_PICASO, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        #self.mu = nn.Parameter(torch.Tensor(1, num_seeds, 1)).cuda()
        #nn.init.xavier_uniform_(self.mu)
        #self.sigma = nn.Parameter(torch.Tensor(1, num_seeds, 1)).cuda()
        #nn.init.xavier_uniform_(self.sigma)
        #self.dim = dim
        #self.num_seeds = num_seeds
        self.mab = MAB(dim, dim_in, dim, num_heads, ln=ln)#, Vis=Vis)
        #self.mab0 = MAB(dim, dim_in, dim, num_heads, ln=ln)#, Vis=Vis)
        #self.mab1 = MAB(dim, dim_in, dim, num_heads, ln=ln)#, Vis=Vis)
        #self.mab2 = MAB(dim, dim_in, dim, num_heads, ln=ln)#, Vis=Vis)
        #self.mab3 = MAB(dim, dim_in, dim, num_heads, ln=ln)#, Vis=Vis)
        #self.mab4 = MAB(dim, dim_in, dim, num_heads, ln=ln)#, Vis=Vis)
        #self.mab5 = MAB(dim, dim_in, dim, num_heads, ln=ln, Vis=Vis)
        #self.mab6 = MAB(dim, dim_in, dim, num_heads, ln=ln, Vis=Vis)
        #self.mab = MAB(dim, dim, dim, num_heads, ln=ln)
        self.Gamma = nn.Linear(dim_in, dim)
        self.Lambda = nn.Linear(dim, dim, bias=False)


    def forward(self, X):
        
        H = self.mab(self.S.repeat(X.size(0), 1, 1), X)
        X_prime = self.mab(X, H)

        H_prime = self.mab(H, X_prime)
        X_prime2 = self.mab(X_prime, H_prime)

        H_prime2 = self.mab(H_prime, X_prime2)
        X_prime3 = self.mab(X_prime2, H_prime2)

        #H_prime3 = self.mab(H_prime2, X_prime3)
        #X_prime4 = self.mab(X_prime3, H_prime3)
        
        xm = self.Lambda(H_prime2)
        x = self.Gamma(X_prime3)
        O = x - xm
        return O
