"""adapted from:
https://github.com/juho-lee/set_transformer
"""

from modules import *
from classifier import *

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=True):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                AE(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                AE(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        )
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SA(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))


class PicasoModel(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=True):
        super(PicasoModel, self).__init__()
        self.enc = nn.Sequential(
                AE(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                AE(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        )
        self.dec = nn.Sequential(
                PICASO(dim_hidden, num_heads, num_outputs, ln=ln),
                SA(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))
