import time
from wavenet_model import WaveNetModel, ExpandingWaveNetModel
from wavenet_modules import Conv1dExtendable
from wavenet_training import *
from torch.autograd import Variable
import torch
import numpy as np

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv1dExtendable(in_channels=1,
                                      out_channels=4,
                                      kernel_size=1,
                                      bias=False)
        self.conv2 = Conv1dExtendable(in_channels=1,
                                      out_channels=4,
                                      kernel_size=1,
                                      bias=False)
        self.conv3 = Conv1dExtendable(in_channels=4,
                                      out_channels=4,
                                      kernel_size=1,
                                      bias=False)
        self.conv4 = Conv1dExtendable(in_channels=4,
                                      out_channels=2,
                                      kernel_size=1,
                                      bias=True)
        self.conv1.input_tied_modules = [self.conv3]
        self.conv1.output_tied_modules = [self.conv2]
        self.conv2.input_tied_modules = [self.conv3]
        self.conv2.output_tied_modules = [self.conv1]
        self.conv3.input_tied_modules = [self.conv4]

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = selu(x1 + x2)
        x = selu(self.conv3(x))
        x = selu(self.conv4(x))
        return x


model = Net()

input = Variable(torch.rand(1, 1, 4) * 2 - 1)
output = model(input)
print("output: ", output)

model.conv1.split_feature(feature_number=1)
model.conv2.split_feature(feature_number=3)
#model.conv3.split_feature(feature_number=2)

output2 = model(input)
print("output 2: ", output2)

diff = output - output2
dot = torch.dot(diff, diff)
print("mse: ", dot.data[0])
