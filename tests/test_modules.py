import torch
from torch.autograd import Variable
from unittest import TestCase
from model import ConvDilated, dilate

class Test_Dilation(TestCase):
	def test_dilate(self):
		input = Variable(torch.linspace(0, 12, steps=13).view(1, 1, 13))

		dilated = dilate(input, 1)
		assert dilated.size() == (1, 1, 13)
		assert dilated[0, 0, 4] == 4
		print(dilated)

		dilated = dilate(input, 2)
		assert dilated.size() == (2, 1, 7)
		assert dilated[1, 0, 2] == 4
		print(dilated)

		dilated = dilate(dilated, 4)
		assert dilated.size() == (4, 1, 4)
		assert dilated[3, 0, 1] == 4
		print(dilated)

		dilated = dilate(dilated, 1)
		assert dilated.size() == (1, 1, 16)
		assert dilated[0, 0, 7] == 4
		print(dilated)

	def test_dilate_multichannel(self):
		input = Variable(torch.linspace(0, 35, steps=36).view(2, 3, 6))

		dilated = dilate(input, 1)
		dilated = dilate(input, 2)
		dilated = dilate(input, 4)



class Test_ConvDilated(TestCase):
	def test_dilation(self):
		module = ConvDilated(num_channels_in=1,
							 num_channels_out=1,
							 kernel_size=2,
							 dilation=2)

		module.conv.weight = torch.nn.Parameter(torch.FloatTensor([[[0, 1]]]))
		w = module.conv.weight

		input = Variable(torch.linspace(0, 12, steps=13).view(1, 1, 13))
		dilated = module(input)
		print(dilated)
		assert dilated.size() == (2, 1, 6)
		res = torch.FloatTensor([[[1, 3, 5, 7,  9, 11]],
								 [[2, 4, 6, 8, 10, 12]]])
		assert dilated.data.equal(res)

		module.dilation = 4
		dilated = module(dilated)
		print(dilated)
		assert dilated.size() == (4, 1, 2)
		res = torch.FloatTensor([[[5,  9]],
								 [[6, 10]],
								 [[7, 11]],
								 [[8, 12]]])
		assert dilated.data.equal(res)

		module.dilation = 1
		dilated = module(dilated)
		print(dilated)
		assert dilated.size() == (1, 1, 7)
		res = torch.FloatTensor([[[6, 7, 8, 9, 10, 11, 12]]])
		assert dilated.data.equal(res)
