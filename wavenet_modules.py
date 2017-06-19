import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable, Function

import numpy as np

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha)


class WaveNetLayer(nn.Module):
    r"""Base module of the WaveNet architecture. Applies dilation and a 1D convolution over a multi-channel input signal.
        Allows optional residual connection if the number of input channels equals the number of output channels.

        Args:
            in_channels (int): Number of input channels. Should be the size of the second dimension of the input tensor
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int): Size of the convolution kernel
            dilation (int): Target dilation, affects dimensions 0 and 2 of the input tensor
            dilation_init (int): Initial dilation of the input
            residual_connection (bool): If true, the dilated input will be added to the output of the convolution

        Shape:
            - Input: :math:`N_{in}, C_{in}, L_{in}`
            - Output: :math:`N_{out}, C_{out}, L_{out}` where
              :math:`d = dilation / dilation_init`, :math:`N_{out} = floor(N_{in}*d), L_{out} = floor(L_{in} / d)`
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 dilation=2,
                 init_dilation=1,
                 residual_connection=True):

        super(WaveNetLayer, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.init_dilation = init_dilation
        self.residual = residual_connection

        self.dil_conv = nn.Conv1d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  bias=False)

        self.onexone_conv = nn.Conv1d(in_channels=out_channels,
                                      out_channels=out_channels,
                                      kernel_size=1,
                                      bias=False)

        self.queue = DilatedQueue(max_length=(kernel_size - 1) * dilation + 1,
                                  num_channels=in_channels,
                                  dilation=dilation)

    def forward(self, input):

        #		|
        # +----add
        # |		|
        # |	   1x1
        # |		|
        # |	   ReLU
        # |		|
        # |	   dil_conv
        # |		|
        # |	   ReLU
        # +-----|

        input = dilate(input,
                       dilation=self.dilation,
                       init_dilation=self.init_dilation)
        r = F.relu(input)

        # zero padding for convolution
        l = r.size(2)
        if l < self.kernel_size:
            r = constant_pad_1d(r, self.kernel_size - l, dimension=2, pad_start=True)

        r = self.dil_conv(r)
        r = F.relu(r)
        r = self.onexone_conv(r)

        if self.residual:
            input = input[:, :, (self.kernel_size - 1):]
            r = r + input.expand_as(r)

        return r

    def generate(self, new_sample):
        self.queue.enqueue(new_sample)
        input = self.queue.dequeue(num_deq=self.kernel_size,
                                   dilation=self.dilation)

        input = input.unsqueeze(0)
        # input = Variable(input, volatile=True)

        r = F.relu(input)
        r = self.dil_conv(r)
        r = F.relu(r)
        r = self.onexone_conv(r)

        if self.residual:
            input = input[:, :, (self.kernel_size - 1):]
            r = r + input.expand_as(r)

        return r


class DilatedConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 dilation=2,
                 init_dilation=1):
        super(DilatedConv, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.init_dilation = init_dilation

        self.dil_conv = nn.Conv1d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  bias=False)

        self.queue = DilatedQueue(max_length=(kernel_size - 1) * dilation + 1,
                                  num_channels=in_channels,
                                  dilation=dilation)

    def forward(self, input):
        r = dilate(input,
                   dilation=self.dilation,
                   init_dilation=self.init_dilation)

        # zero padding for convolution
        l = r.size(2)
        if l < self.kernel_size:
            r = constant_pad_1d(r, self.kernel_size - l, dimension=2, pad_start=True)

        r = self.dil_conv(r)

        return r

    def generate(self, new_sample):
        self.queue.enqueue(new_sample)
        input = self.queue.dequeue(num_deq=self.kernel_size,
                                   dilation=self.dilation)

        r = input.unsqueeze(0)
        return r


class WaveNetFinalLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 out_length,
                 init_dilation):
        super(WaveNetFinalLayer, self).__init__()

        self.out_length = out_length
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=num_classes,
                              kernel_size=1,
                              bias=True)
        self.init_dilation = init_dilation

    def forward(self, input):
        input = dilate(input,
                       dilation=1,
                       init_dilation=self.init_dilation)
        r = F.relu(input)
        r = self.conv(r)

        # reshape
        [n, c, l] = r.size()
        r = r.transpose(1, 2).contiguous().view(n * l, c)
        r = r[-self.out_length * n:, :]
        return r

    def generate(self, new_sample):
        # r = new_sample.unsqueeze(0)
        r = self.conv(new_sample)
        # r = self.conv(Variable(r, volatile=True))
        r = r.data.squeeze()
        return r


def dilate(x, dilation, init_dilation=1, pad_start=True):
    """
    :param x: Tensor of size (N, C, L), where N is the input dilation, C is the number of channels, and L is the input length
    :param dilation: Target dilation. Will be the size of the first dimension of the output tensor.
    :param pad_start: If the input length is not compatible with the specified dilation, zero padding is used. This parameter determines wether the zeros are added at the start or at the end.
    :return: The dilated tensor of size (dilation, C, L*N / dilation). The output might be zero padded at the start
    """

    [n, c, l] = x.size()
    dilation_factor = dilation / init_dilation
    if dilation_factor == 1:
        return x

    # zero padding for reshaping
    new_l = int(np.ceil(l / dilation_factor) * dilation_factor)
    if new_l != l:
        l = new_l
        x = constant_pad_1d(x, new_l, dimension=2, pad_start=pad_start)

    l = int(round(l / dilation_factor))
    n = int(round(n * dilation_factor))

    # reshape according to dilation
    x = x.permute(1, 2, 0).contiguous()  # (n, c, l) -> (c, l, n)
    x = x.view(c, l, n)
    x = x.permute(2, 0, 1).contiguous()  # (c, l, n) -> (n, c, l)

    return x


class DilatedQueue:
    def __init__(self, max_length, data=None, dilation=1, num_deq=1, num_channels=1, dtype=torch.FloatTensor):
        self.in_pos = 0
        self.out_pos = 0
        self.num_deq = num_deq
        self.num_channels = num_channels
        self.dilation = dilation
        self.max_length = max_length
        self.data = data
        self.dtype = dtype
        if data == None:
            self.data = Variable(dtype(num_channels, max_length).zero_())

    def enqueue(self, input):
        self.data[:, self.in_pos] = input
        self.in_pos = (self.in_pos + 1) % self.max_length

    def dequeue(self, num_deq=1, dilation=1):
        #       |
        #  |6|7|8|1|2|3|4|5|
        #         |
        start = self.out_pos - ((num_deq - 1) * dilation)
        if start < 0:
            t1 = self.data[:, start::dilation]
            t2 = self.data[:, self.out_pos % dilation:self.out_pos + 1:dilation]
            t = torch.cat((t1, t2), 1)
        else:
            t = self.data[:, start:self.out_pos + 1:dilation]

        self.out_pos = (self.out_pos + 1) % self.max_length
        return t

    def reset(self):
        self.data = Variable(self.dtype(self.num_channels, self.max_length).zero_())
        self.in_pos = 0
        self.out_pos = 0


class ConstantPad1d(Function):
    def __init__(self, target_size, dimension=0, value=0, pad_start=False):
        super(ConstantPad1d, self).__init__()
        self.target_size = target_size
        self.dimension = dimension
        self.value = value
        self.pad_start = pad_start

    def forward(self, input):
        self.num_pad = self.target_size - input.size(self.dimension)
        assert self.num_pad >= 0, 'target size has to be greater than input size'

        self.input_size = input.size()

        size = list(input.size())
        size[self.dimension] = self.target_size
        output = input.new(*tuple(size)).fill_(self.value)
        c_output = output

        # crop output
        if self.pad_start:
            c_output = c_output.narrow(self.dimension, self.num_pad, c_output.size(self.dimension) - self.num_pad)
        else:
            c_output = c_output.narrow(self.dimension, 0, c_output.size(self.dimension) - self.num_pad)

        c_output.copy_(input)
        return output

    def backward(self, grad_output):
        grad_input = grad_output.new(*self.input_size).zero_()
        cg_output = grad_output

        # crop grad_output
        if self.pad_start:
            cg_output = cg_output.narrow(self.dimension, self.num_pad, cg_output.size(self.dimension) - self.num_pad)
        else:
            cg_output = cg_output.narrow(self.dimension, 0, cg_output.size(self.dimension) - self.num_pad)

        grad_input.copy_(cg_output)
        return grad_input


class Conv1dExtendable(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(Conv1dExtendable, self).__init__(*args, **kwargs)
        self.init_ncc()
        self.input_tied_modules = [] # modules whose input is sensitive to the output size of this module
        self.output_tied_modules = [] # modules whose output size has to be compatible this this modules output

    def init_ncc(self):
        w = self.weight.view(self.weight.size(0), -1)  # size: (G, F*J)
        mean = torch.mean(w, dim=1).unsqueeze(1).expand_as(w)
        self.t0_factor = w - mean
        self.t0_norm = torch.norm(w, p=2, dim=1)
        self.start_ncc = Variable(torch.zeros(self.out_channels))
        self.start_ncc = self.normalized_cross_correlation()

    def normalized_cross_correlation(self):
        w = self.weight.view(self.weight.size(0), -1)
        t_norm = torch.norm(w, p=2, dim=1)

        # If there is only one input channel, no sensible ncc can be calculated, return instead the ratio of the norms
        if self.in_channels == 1 & sum(self.kernel_size) == 1:
            ncc = w.squeeze() / torch.norm(self.t0_norm, 2)
            ncc = ncc - self.start_ncc
            return ncc
        mean = torch.mean(w, dim=1).unsqueeze(1).expand_as(w)
        t_factor = w - mean
        h_product = self.t0_factor * t_factor
        covariance = torch.sum(h_product, dim=1) #/ (w.size(1)-1)

        #t_sd = torch.std(w, dim=1)
        #normalization_factor = 1 / (self.t0_sd * t_sd)

        denominator = self.t0_norm * t_norm

        ncc = covariance / denominator
        ncc = ncc - self.start_ncc
        return ncc

    def split_feature(self, feature_number):
        self.split_output_channel(channel_number=feature_number)
        for dep in self.input_tied_modules:
            dep.split_input_channel(channel_number=feature_number)
        for dep in self.output_tied_modules:
            dep.split_output_channel(channel_number=feature_number)

    def split_features(self, threshold):
        ncc = self.normalized_cross_correlation()
        for i, ncc_value in enumerate(ncc):
            if ncc_value < threshold:
                print("ncc value for feature ", i, ": ", ncc_value)
                self.split_feature(i)

    def split_output_channel(self, channel_number):
        '''
        Split one output channel (a feature) in two, but retain the same summed value

        :param channel_number: The number of the channel that will be split
        '''

        # weight tensor: (out_channels, in_channels, kernel_size)
        self.out_channels += 1

        original_weight = self.weight.data
        split_positions = 2 * torch.rand(self.in_channels, self.kernel_size[0])

        new_weight = torch.zeros(self.out_channels, self.in_channels, self.kernel_size[0])
        if channel_number > 0:
            new_weight[0:channel_number, :, :] = original_weight[0:channel_number, :, :]
        new_weight[channel_number, :, :] = original_weight[channel_number, :, :] * split_positions
        new_weight[channel_number + 1, :, :] = original_weight[channel_number, :, :] * (2 - split_positions)
        if channel_number + 2 < self.out_channels:
            new_weight[channel_number + 2:, :, :] = original_weight[channel_number + 1:, :, :]

        if self.bias is not None:
            original_bias = self.bias.data
            new_bias = torch.zeros(self.out_channels)
            new_bias[0:channel_number+1] = original_bias[0:channel_number+1]
            new_bias[channel_number+1:] = original_bias[channel_number:]
            self.bias = Parameter(new_bias)

        self.weight = Parameter(new_weight)
        self.init_ncc()

    def split_input_channel(self, channel_number):

        if channel_number > self.in_channels:
            print("cannot split in channel ", channel_number)
            return

        self.in_channels += 1
        original_weight = self.weight.data
        duplicated_slice = original_weight[:, channel_number, :] * 0.5

        new_weight = torch.zeros(self.out_channels, self.in_channels, self.kernel_size[0])
        if channel_number > 0:
            new_weight[:, 0:channel_number, :] = original_weight[:, 0:channel_number, :]
        new_weight[:, channel_number, :] = duplicated_slice
        new_weight[:, channel_number+1, :] = duplicated_slice
        if channel_number+2 < self.in_channels:
            new_weight[:, channel_number+2:, :] = original_weight[:, channel_number+1:, :]
        self.weight = Parameter(new_weight)
        self.init_ncc()



def constant_pad_1d(input,
                    target_size,
                    dimension=0,
                    value=0,
                    pad_start=False):
    return ConstantPad1d(target_size, dimension, value, pad_start)(input)


def mu_law_enconding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x


def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s
