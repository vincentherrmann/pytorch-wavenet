import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable, Function
import numpy as np


def dilate(x, dilation, init_dilation=1, pad_start=True):
    """
    :param x: Tensor of size (N, C, L), where N is the input dilation, C is the number of channels, and L is the input length
    :param dilation: Target dilation. Will be the size of the first dimension of the output tensor.
    :param pad_start: If the input length is not compatible with the specified dilation, zero padding is used. This parameter determines wether the zeros are added at the start or at the end.
    :return: The dilated tensor of size (dilation, C, L*N / dilation). The output might be zero padded at the start
    """

    [n, c, l] = x.size()
    if n == 1 and l == 1:
        return x
    dilation_factor = dilation / init_dilation
    if dilation_factor == 1:
        return x

    # zero padding for reshaping
    new_l = int(np.ceil(l / dilation_factor) * dilation_factor)
    if new_l != l:
        l = new_l
        x = constant_pad_1d(x, new_l, dimension=2, pad_start=pad_start)

    # l_old = int(round(l / dilation_factor))
    # n_old = int(round(n * dilation_factor))
    l = math.ceil(l * init_dilation / dilation)
    n = math.ceil(n * dilation / init_dilation)

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


def constant_pad_1d(input,
                    target_size,
                    dimension=0,
                    value=0,
                    pad_start=False):
    return ConstantPad1d(target_size, dimension, value, pad_start)(input)


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis  = len(x.size()) - 2
    m, _  = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return Variable(one_hot)


def mix_logistic_log_probability(parameters, samples, bin_size=0., reduce=True):
    if len(samples.size()) == 1:
        samples = samples.unsqueeze(1)
        num_samples = 1
    else:
        num_samples = samples.size(1)
    samples = samples.unsqueeze(1)
    nr_mix = parameters.size()[-1] // 3  # number of mixtures, // 3 because we have weights, means and scales

    # parameters of the mixtures
    weights = parameters[:, :nr_mix]
    means = parameters[:, nr_mix:2 * nr_mix]
    log_scales = torch.clamp(parameters[:, 2 * nr_mix:], min=-7.)  # clamp for numerical stability

    # calculate the probabilities for each distribution (see equation (2) in the PixelCNN++ paper)
    distances_to_target = samples - means.unsqueeze(2)
    inv_scales = torch.exp(-log_scales).unsqueeze(2)
    pos_in = inv_scales * (distances_to_target + bin_size)
    neg_in = inv_scales * (distances_to_target - bin_size)
    pos_cdf = F.sigmoid(pos_in)
    neg_cdf = F.sigmoid(neg_in)
    cdf_delta = pos_cdf - neg_cdf  # the regular probability

    log_distributions = torch.log(torch.clamp(cdf_delta, min=1e-12))
    weighted_log_distributions = log_distributions + log_prob_from_logits(weights).unsqueeze(2)
    log_probabilities = log_sum_exp(weighted_log_distributions)
    if num_samples == 1:
        log_probabilities = log_probabilities.squeeze()
    else:
        log_probabilities = torch.sum(log_probabilities, dim=2) / num_samples
    if reduce:
        return -torch.sum(log_probabilities)
    else:
        return -log_probabilities


def discretized_mix_logistic_loss(input, target, bin_count=0, reduce=True):
    """

    :param input: (minibatch, P)
    :param target: (minibatch, N), second dimension is an optional sample dimension. Should be scaled to [-1, 1]
    :param bin_count:
    :return:
    """

    if len(target.size()) == 1:
        target = target.unsqueeze(1)
        num_samples = 1
    else:
        num_samples = target.size(1)
    nr_mix = input.size()[-1] // 3  # number of mixtures, // 3 because we have weights, means and scales
    target = target.unsqueeze(1)

    # parameters of the mixtures
    weights = input[:, :nr_mix]
    means = input[:, nr_mix:2*nr_mix]
    log_scales = torch.clamp(input[:, 2*nr_mix:], min=-7.).unsqueeze(2)  # clamp for numerical stability

    # calculate the probabilities for each distribution (see equation (2) in the PixelCNN++ paper)
    distances_to_target = target - means.unsqueeze(2)
    inv_scales = torch.exp(-log_scales)

    mid_in = inv_scales * distances_to_target
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)  # edge case 1; very low probabilities

    if bin_count <= 1:
        # if the bin count is 0 or 1, calculate the continuous probability
        out = log_pdf_mid + math.log(2) # add factor to get to normalized log probabilities (why??)
    else:
        # calculate the log probability as the delta between two CDFs displaced by bin_size
        pos_in = inv_scales * (distances_to_target + 1/bin_count)
        neg_in = inv_scales * (distances_to_target - 1/bin_count)
        pos_cdf = F.sigmoid(pos_in)
        neg_cdf = F.sigmoid(neg_in)
        cdf_delta = pos_cdf - neg_cdf  # the regular probability

        condition = (cdf_delta > 1e-5).float()  # probability large enough
        out = condition * torch.log(torch.clamp(cdf_delta, min=1e-12)) \
            + (1. - condition) * (log_pdf_mid - np.log((bin_count - 1) / 2.))

        # consider edge cases
        log_one_minus_cdf_neg = -F.softplus(neg_in)  # case target == 1
        log_cdf_pos = pos_in - F.softplus(pos_in)  # case target == -1

        # compose conditions
        condition = (target > 0.999).float()  # target == 1
        out = condition * log_one_minus_cdf_neg \
              + (1. - condition) * out
        condition = (target < -0.999).float()  # target == -1
        out = condition * log_cdf_pos \
              + (1. - condition) * out  # (N, C, M)

    # weigh the log probabilities and add them together
    log_probabilities = out + log_prob_from_logits(weights).unsqueeze(2)
    combined_log_probabilities = log_sum_exp(log_probabilities)
    combined_log_probabilities = torch.sum(combined_log_probabilities, dim=1) / num_samples
    if reduce:
        return -torch.sum(combined_log_probabilities)
    else:
        return -combined_log_probabilities


def get_modes_from_discretized_mix_logistic(parameters, bin_count=256):
    """
    get the single bin with the highest probability (or lowest loss) from the distribution
    :param parameters:
    :param bin_count:
    :return:
    """
    nr_mix = parameters.size()[-1] // 3  # number of mixtures, // 3 because we have weights, means and scales
    means = parameters[:, nr_mix:2 * nr_mix]
    losses = Variable(torch.FloatTensor(parameters.size(0), nr_mix), volatile=True)
    if parameters.is_cuda:
        losses = losses.cuda()

    # calculate the loss at each mean position
    for m in range(nr_mix):
        losses[:, m] = discretized_mix_logistic_loss(parameters,
                                                     target=means[:, m],
                                                     bin_count=bin_count,
                                                     reduce=False)

    # select the one with the lowest loss
    _, argmin = torch.min(losses, dim=1)
    selection = argmin.unsqueeze(1)
    modes = torch.gather(means, dim=1, index=selection)
    return modes.squeeze()


def sample_from_discretized_mix_logistic(parameters, temperature=1.0):
    """

    :param parameters: (batch, P)
    :param temperature: (float)
    :return: (batch)
    """

    nr_mix = parameters.size()[-1] // 3  # number of mixtures, // 3 because we have weights, means and scales

    # parameters of the mixtures
    weights = parameters[:, :nr_mix]
    means = parameters[:, nr_mix:2 * nr_mix]
    temp_log = math.log(temperature)
    log_scales = torch.clamp(parameters[:, 2 * nr_mix:] + temp_log, min=-7.)  # clamp for numerical stability

    # sample mixture indicator from softmax
    temp = torch.FloatTensor(weights.size())
    if parameters.is_cuda:
        temp = temp.cuda()
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = weights.data - torch.log(-torch.log(temp))  # weigh the individual distributions
    _, argmax = temp.max(dim=1)  # select the distribution from which we will sample
    selection = Variable(argmax, volatile=True).unsqueeze(1)

    means = torch.gather(means, dim=1, index=selection)
    log_scales = torch.gather(log_scales, dim=1, index=selection)

    u = torch.FloatTensor(means.size())
    if parameters.is_cuda:
        u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    u = Variable(u)

    # sample from the logistic distribution using the corresponding quantile function
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x = torch.clamp(torch.clamp(x, min=-1.), max=1.)
    return x


def sample_from_softmax(x, temperature=1., bin_count=256):
    x /= temperature
    prob = F.softmax(x, dim=0)
    prob = prob.cpu()
    np_prob = prob.data.numpy()
    x = np.random.choice(bin_count, p=np_prob)
    x = np.array([x])
    x = (x / bin_count) * 2. - 1
    return x


def sample_from_mixture(x, temperature=1., bin_count=256):
    x = sample_from_discretized_mix_logistic(x.unsqueeze(0), temperature=temperature)
    x = x.cpu().data
    #x = int(((x+1.)*0.5) * bin_count)
    #x = (x / bin_count) * 2. - 1.
    return np.array([x])
