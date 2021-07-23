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


        '''A_ = A.view(Q.size(0), A.size(1), -1)
        #alpha = A_.repeat(1, 25)
        if self.Vis:
            image_path = '/home/cougarnet.uh.edu/szare836/Desktop/04/000659_04.jpg'
            image = Image.open(image_path)
            raw = image
            image = transforms.Resize(48)(image)
            image = transforms.CenterCrop(32)(image)

            A_ = A[0, :, :]
            alpha = A_.squeeze(0).unsqueeze(1).repeat(1,16)
            alpha = alpha.cpu().detach().numpy()
            alpha = alpha.reshape(64, 4, 4)
            s = (32,32)
            a0 = np.zeros(s)
            num = 0
            for i in range(8):
                for j in range(8):
                    a0[4 * i:4 * i + 4, 4 * j:4 * j + 4] = alpha[num, :, :]
                    num = num+1

            A_ = A[1, :, :]
            alpha = A_.squeeze(0).unsqueeze(1).repeat(1, 16)
            alpha = alpha.cpu().detach().numpy()
            alpha = alpha.reshape(64, 4, 4)
            s = (32, 32)
            a1 = np.zeros(s)
            num = 0
            for i in range(8):
                for j in range(8):
                    a1[4 * i:4 * i + 4, 4 * j:4 * j + 4] = alpha[num, :, :]
                    num = num + 1

            A_ = A[2, :, :]
            alpha = A_.squeeze(0).unsqueeze(1).repeat(1, 16)
            alpha = alpha.cpu().detach().numpy()
            alpha = alpha.reshape(64, 4, 4)
            s = (32, 32)
            a2 = np.zeros(s)
            num = 0
            for i in range(8):
                for j in range(8):
                    a2[4 * i:4 * i + 4, 4 * j:4 * j + 4] = alpha[num, :, :]
                    num = num + 1

            A_ = A[3, :, :]
            alpha = A_.squeeze(0).unsqueeze(1).repeat(1, 16)
            alpha = alpha.cpu().detach().numpy()
            alpha = alpha.reshape(64, 4, 4)
            s = (32, 32)
            a3 = np.zeros(s)
            num = 0
            for i in range(8):
                for j in range(8):
                    a3[4 * i:4 * i + 4, 4 * j:4 * j + 4] = alpha[num, :, :]
                    num = num + 1


            fig, ax = plt.subplots(1, 5)
            ax[0].imshow(image, cmap='gray')
            ax[0].axis('off')

            ax[1].imshow(image, cmap='gray')
            ax[1].imshow(a0, cmap='jet', alpha=0.5)
            ax[1].axis('off')

            ax[2].imshow(image, cmap='gray')
            ax[2].imshow(a1, cmap='jet', alpha=0.5)
            ax[2].axis('off')

            ax[3].imshow(image, cmap='gray')
            ax[3].imshow(a2, cmap='jet', alpha=0.5)
            ax[3].axis('off')

            ax[4].imshow(image, cmap='gray')
            ax[4].imshow(a3, cmap='jet', alpha=0.5)
            ax[4].axis('off')
            #plt.show()
            plt.savefig('line_plot.png', bbox_inches='tight', pad_inches=0, dpi=500)'''

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
