from unittest import TestCase
from wavenet_model import *
import torch
from torch.autograd.variable import Variable


class TestWaveNetModel(TestCase):
    def test_SingleBlockWaveNet(self):
        model = WaveNetModel(layers=4,
                             blocks=1,
                             dilation_channels=16,
                             residual_channels=16,
                             skip_channels=32,
                             end_channels=64,
                             classes=256,
                             output_length=8)

        test_input = torch.zeros(4, 256, model.receptive_field + model.output_length - 1)
        test_input = Variable(test_input)
        result = model(test_input)