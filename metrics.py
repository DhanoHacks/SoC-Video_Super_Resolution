# This file contains the metric (PSNR) used to statistically evaluate the performance of model

import torch
from torch import nn

def calc_psnr(img1, img2, max_val=1.0):
    MSE = nn.MSELoss()
    return 10 * torch.log10(max_val * max_val / MSE(img1, img2))