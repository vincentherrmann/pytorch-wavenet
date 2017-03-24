import torch
from torch.autograd import Variable
from unittest import TestCase
from model import ConvDilated

class Test_ConvDilated(TestCase):
	def test_dilation(self):
		module = ConvDilated(num_channels_in=1,
							 num_channels_out=1,
							 kernel_size=2,
							 dilation=2)

		input = Variable(torch.linspace(0, 12, steps=13).view(1, 1, 13))
		dilated = module.dilate(input)
		assert dilated.size() == (2, 1, 7)
		assert dilated[1, 0, 2] == 4
		print(dilated)

		module.dilation = 4
		dilated = module.dilate(dilated)
		assert dilated.size() == (4, 1, 4)
		assert dilated[3, 0, 1] == 4
		print(dilated)

		module.dilation = 1
		dilated = module.dilate(dilated)
		assert dilated.size() == (1, 1, 16)
		assert dilated[0, 0, 7] == 4
		print(dilated)
