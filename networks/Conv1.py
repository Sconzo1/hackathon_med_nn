import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Conv1Net(nn.Module):
    def __init__(self):
        super(Conv1Net, self).__init__()
                                                                                                        #N, 8, 2000
        self.conv1 = nn.Conv1d(kernel_size=7, in_channels=8, out_channels=32, padding=7//2, stride=1)   #N, 32, 2000
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)                                   #N, 32, 1000
        self.conv2 = nn.Conv1d(kernel_size=7, in_channels=32, out_channels=64, padding=7//2, stride=1)  #N, 64, 1000
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)                                   #N, 64, 500
        self.conv3 = nn.Conv1d(kernel_size=7, in_channels=64, out_channels=128, padding=7//2, stride=1) #N, 128, 500
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)                                   #N, 128, 250
        self.conv4 = nn.Conv1d(kernel_size=7, in_channels=128, out_channels=64, padding=7//2, stride=2) #N, 64, 125
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)                                   #N, 64, 62

        self.fc1 = nn.Linear(in_features=64*62, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=3)

    """
    x - Tensor[batch_size, 8, 2000]
    """
    def forward(self, x):
        x = self.pool1( F.relu( self.conv1(x) ) )
        x = self.pool2( F.relu( self.conv2(x) ) )
        x = self.pool3( F.relu( self.conv3(x) ) )
        x = self.pool4( F.relu( self.conv4(x) ) )

        x = x.view(-1, 64*62)
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )
        x = F.log_softmax( self.fc3(x), dim = 1 )
        return x
