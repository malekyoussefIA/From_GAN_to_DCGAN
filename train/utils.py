
import torch.nn as nn
import torch


""" 
The authors in the DCGAN paper specify that all model weights shall be  initialized 
from a Normal distribution with a of '0' and a  Standard Deviations of '0.02'
To do that we will use a function taken from this Pytorch tutorial : https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#discriminator
"""

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


"Function that returns noise for generator input"
def get_noise(img_size,noise_dim):
    
    "Function that returns noise for generator input"
    return torch.randn(img_size, noise_dim, 1, 1)

