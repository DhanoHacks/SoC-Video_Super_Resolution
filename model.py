# Contains the neural network


from torch import nn
from torch import clip
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_channels=1):
        super(Net, self).__init__() #size=[4,1,33,33]
        self.conv1 = nn.Conv2d(1,64,9)
        self.conv2 = nn.Conv2d(64,32,1)
        self.recon = nn.Conv2d(32,1,5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.recon(x)
        x_out = clip(x, 0.0, 1.0)
        return x_out #final size [4,1,21,21]