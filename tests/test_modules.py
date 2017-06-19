import torch
from torch.autograd import Variable
from unittest import TestCase
from wavenet_training import dilate, Conv1dExtendable


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

class Test_Conv1dExtendable:
    def test_ncc(self):
        module = Conv1dExtendable(in_channels=3,
                                  out_channels=5,
                                  kernel_size=4)

        rand = Variable(torch.rand(5, 3, 4))
        module._parameters['weight'] = module.weight * module.weight + rand * 1
        ncc = module.normalized_cross_correlation()
        print(ncc)