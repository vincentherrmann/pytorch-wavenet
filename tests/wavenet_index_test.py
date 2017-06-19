import time
from wavenet_model import WaveNetModel
from wavenet_training import *
from torch.autograd import Variable
import torch
import numpy as np

model = WaveNetModel(layers=7,
                     blocks=2,
                     dilation_channels=1,
                     residual_channels=1,
                     skip_channels=1,
                     classes=256,
                     output_length=1)

# set start convolutions to 1:
model._modules['start_conv']._parameters['weight'].data = model._modules['start_conv']._parameters['weight'].data * 0 + 1

# set residual convolutions to 0:
for c in model.residual_convs:
    c._parameters['weight'].data.zero_()

# set skip  convolutions to 0:
for c in model.skip_convs:
    c._parameters['weight'].data = c._parameters['weight'].data * 0 + 1

# set dilated convolutions to [0, 1]:
for c in model.filter_convs:
    c._parameters['weight'].data[:, :, 0] = torch.zeros((model.dilation_channels*2, model.dilation_channels))
    c._parameters['weight'].data[:, :, 1] = torch.zeros((model.dilation_channels*2, model.dilation_channels)) + 1.

input = np.linspace(0, model.receptive_field-1, num=model.receptive_field, dtype=np.float32)
input = torch.from_numpy(input)
model(Variable(input.view(1, 1, -1)))
