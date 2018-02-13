from unittest import TestCase
import torch
from wavenet_modules import *
from torch.autograd import Variable

if __name__ == '__main__':
    try:
        raise Exception()
    except:
        import pdb
        pdb.set_trace()

class TestMixLogistic(TestCase):
    def test_sample_from_discretized_mix_logistic(self):
        parameters = torch.zeros(3, 6).uniform_(-1., 1.)
        # parameters = torch.FloatTensor([[2., -1.1, -0.3, 0.6, 0.9, -3.], [2., -1.1, -0.3, 0.6, 0.9, -3.], [2., -1.1, -0.3, 0.6, 0.9, -3.]])
        sample = sample_from_discretized_mix_logistic(Variable(parameters, volatile=True))

    def test_discretized_mix_logistic_loss(self):
        targets = torch.linspace(-1., 1., 100).unsqueeze(1)
        parameters = torch.zeros(1, 3).uniform_(-1., 1.)
        for t in targets:
            loss = discretized_mix_logistic_loss(Variable(parameters), target=Variable(t))
            print(loss)
