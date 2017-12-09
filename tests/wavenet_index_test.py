from unittest import TestCase
import time
from wavenet_model import WaveNetModel
#from wavenet_training import *
from torch.autograd import Variable
import torch
import numpy as np

class TestWaveNetModel(TestCase):
    def setUp(self):
        self.model = WaveNetModel(layers=3,
                             blocks=2,
                             dilation_channels=4,
                             residual_channels=4,
                             skip_channels=4)

        # set start convolutions to 1:
        self.model._modules['start_conv']._parameters['weight'].data = self.model._modules['start_conv']._parameters[
                                                                      'weight'].data * 0 + 1

        # set residual convolutions to 0:
        for c in self.model.residual_convs:
            c._parameters['weight'].data.zero_()

        # set skip  convolutions to 0:
        for c in self.model.skip_convs:
            c._parameters['weight'].data = c._parameters['weight'].data * 0 + 1

        # set dilated convolutions to [0, 1]:
        for c in self.model.filter_convs:
            c._parameters['weight'].data[:, :, 0] = c._parameters['weight'].data[:, :, 0] * 0
            c._parameters['weight'].data[:, :, 1] = c._parameters['weight'].data[:, :, 0] + 1.

        # set end convolutions to 1:
        self.model._modules['end_conv']._parameters['weight'].data = self.model._modules['end_conv']._parameters[
                                                                      'weight'].data * 0 + 1
        self.model._modules['end_conv']._parameters['bias'].data = self.model._modules['end_conv']._parameters[
                                                                      'bias'].data * 0

    def test_simple_architecture(self):
        self.model.output_length = 8
        in_sig = torch.zeros(1, 1, self.model.receptive_field)
        in_sig[:,:,-3] = 1.
        out_sig = self.model(Variable(in_sig))
        assert False
        pass

