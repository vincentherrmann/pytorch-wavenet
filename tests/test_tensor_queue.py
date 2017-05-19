from unittest import TestCase
from scipy.io import wavfile
import torch
import torch.autograd
from torch.nn._functions.padding import ConstantPad2d
import torch.nn.functional as F
from torch.autograd import Variable, gradcheck

from wavenet_training import DilatedQueue, ConstantPad1d


class Test_dilated_queue(TestCase):
    def test_enqueue(self):
        queue = DilatedQueue(max_length=8, num_channels=3)
        e = torch.zeros((3))
        for i in range(11):
            e = e + 1
            queue.enqueue(e)

        data = queue.data[0, :]
        print('data: ', data)
        assert data[0] == 9
        assert data[2] == 11
        assert data[7] == 8

    def test_dequeue(self):
        queue = DilatedQueue(max_length=8, num_channels=1)
        e = torch.zeros((1))
        for i in range(11):
            e = e + 1
            queue.enqueue(e)

        print('data: ', queue.data)

        for i in range(9):
            d = queue.dequeue(num_deq=3, dilation=2)
            print(d)

        assert d[0][0] == 5
        assert d[0][1] == 7
        assert d[0][2] == 9

    def test_combined(self):
        queue = DilatedQueue(max_length=12, num_channels=1)
        e = torch.zeros((1))
        for i in range(30):
            e = e + 1
            queue.enqueue(e)
            d = queue.dequeue(num_deq=3, dilation=4)
            assert d[0][0] == max(i - 7, 0)


class Test_zero_padding(TestCase):
    def test_end_padding(self):
        x = torch.ones((3, 4, 5))

        p = zero_pad(x, num_pad=5, dimension=0)
        assert p.size() == (8, 4, 5)
        assert p[-1, 0, 0] == 0

        p = zero_pad(x, num_pad=5, dimension=1)
        assert p.size() == (3, 9, 5)
        assert p[0, -1, 0] == 0

        p = zero_pad(x, num_pad=5, dimension=2)
        assert p.size() == (3, 4, 10)
        assert p[0, 0, -1] == 0

    def test_start_padding(self):
        x = torch.ones((3, 4, 5))

        p = zero_pad(x, num_pad=5, dimension=0, pad_start=True)
        assert p.size() == (8, 4, 5)
        assert p[0, 0, 0] == 0

        p = zero_pad(x, num_pad=5, dimension=1, pad_start=True)
        assert p.size() == (3, 9, 5)
        assert p[0, 0, 0] == 0

        p = zero_pad(x, num_pad=5, dimension=2, pad_start=True)
        assert p.size() == (3, 4, 10)
        assert p[0, 0, 0] == 0

    def test_narrowing(self):
        x = torch.ones((2, 3, 4))
        x = x.narrow(2, 1, 2)
        print(x)

        x = x.narrow(0, -1, 3)
        print(x)

        assert False


class Test_wav_files(TestCase):
    def test_wav_read(self):
        data = wavfile.read('trained_generated.wav')[1]
        print(data)
        # [0.1, -0.53125...
        assert False


class Test_padding(TestCase):
    def test_1d(self):
        x = Variable(torch.ones((2, 3, 4)), requires_grad=True)

        pad = ConstantPad1d(5, dimension=0, pad_start=False)

        res = pad(x)
        assert res.size() == (5, 3, 4)
        assert res[-1, 0, 0] == 0

        test = gradcheck(ConstantPad1d, x, eps=1e-6, atol=1e-4)
        print('gradcheck', test)

        # torch.autograd.backward(res, )
        res.backward()
        back = pad.backward(res)
        assert back.size() == (2, 3, 4)
        assert back[-1, 0, 0] == 1

    #
    # pad = ConstantPad1d(5, dimension=1, pad_start=True)
    #
    # res = pad(x)
    # assert res.size() == (2, 5, 4)
    # assert res[0, 4, 0] == 0
    #
    # back = pad.backward(res)
    # assert back.size() == (2, 3, 4)
    # assert back[0, 2, 0] == 1


    def test_2d(self):
        pad = ConstantPad2d((5, 0, 0, 0))
        x = Variable(torch.ones((2, 3, 4, 5)))

        res = pad.forward(x)
        print(res.size())
        assert False
