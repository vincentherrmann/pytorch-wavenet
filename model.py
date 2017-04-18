import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.legacy.nn as legacy
import torch.optim as optim
import numpy as np
from torch.autograd import Variable, Function
from scipy.io import wavfile


class ConvDilated(nn.Module):
	def __init__(self,
				 num_channels_in=1,
				 num_channels_out=1,
				 kernel_size=2,
				 dilation=2):
		super(ConvDilated, self).__init__()

		self.num_channels_in = num_channels_in
		self.num_channels_out = num_channels_out
		self.kernel_size = kernel_size

		self.conv = nn.Conv1d(in_channels=num_channels_in,
							  out_channels=num_channels_out,
							  kernel_size=kernel_size,
							  bias=False)
		#self.batchnorm = nn.BatchNorm1d(num_features=num_channels_out, affine=False)

		self.dilation = dilation

		self.queue = DilatedQueue(max_length=(kernel_size-1) * dilation + 1,
								  num_channels=num_channels_in)

	def forward(self, x):
		#print('x size in: ', x.size())

		x = dilate(x, self.dilation)
		l = x.size(2)

		# zero padding for convolution
		if l < self.kernel_size:
			x = constant_pad_1d(x, self.kernel_size-l, dimension=2, pad_start=True)

		x = self.conv(x)
		#x = self.batchnorm()
		#print('size after conv: ', x.size())

		#print('x size out: ', x.size())
		return F.relu(x)

	def generate(self, new_sample):
		self.queue.enqueue(new_sample)
		x = self.queue.dequeue(num_deq=self.kernel_size,
							   dilation=self.dilation)

		#x = self.conv(Variable(x.unsqueeze(0), volatile=True))
		x = F.conv1d(Variable(x.unsqueeze(0), volatile=True),
				 weight=self.conv.weight)
		#x = self.conv.forward(x.unsqueeze(0))
		#x = self.batchnorm(x)

		x = F.relu(x)
		return x.data.squeeze(0)

def dilate(x, dilation):
	[n, c, l] = x.size()
	dilation_factor = dilation / n
	if dilation == n:
		return x

	# zero padding for reshaping
	new_l = int(np.ceil(l / dilation_factor) * dilation_factor)
	if new_l != l:
		l = new_l
		x = constant_pad_1d(x, new_l, dimension=2, pad_start=True)

	# reshape according to dilation
	x = x.transpose(0, 2).contiguous()
	l = (l * n) // dilation
	n = dilation
	x = x.view(l, c, n).transpose(0, 2).contiguous()

	return x



class Final(nn.Module):
	def __init__(self,
				 in_channels=1,
				 num_classes=256):
		super(Final, self).__init__()
		self.num_classes = num_classes
		self.in_channels = in_channels
		self.conv = nn.Conv1d(in_channels, num_classes, kernel_size=1, bias=True)
		#self.batchnorm = nn.BatchNorm1d(num_classes, affine=False)
		#nn.init.normal(self.conv.weight)
		#self.conv.weight = nn.Parameter(torch.FloatTensor([1]))

	def forward(self, x):
		#x = self.batchnorm(self.conv(x))
		x = dilate(x, 1)
		x = self.conv(x)
		[n, c, l] = x.size()
		x = x.transpose(1, 2).contiguous().view(n*l, c)
		return x #F.softmax(x)

	def generate(self, x):
		#x = self.batchnorm(self.conv(Variable(x.unsqueeze(0), volatile=True)))
		#x = self.conv(Variable(x.unsqueeze(0), volatile=True))
		x = F.conv1d(Variable(x.unsqueeze(0), volatile=True),
					 weight=self.conv.weight,
					 bias=self.conv.bias)
		#x = self.conv.forward(x.unsqueeze(0))
		max_index = torch.max(x.squeeze(), 0)[1]
		s = (max_index.data[0] / self.num_classes) * 2. - 1
		return s


class DilatedQueue:
	def __init__(self, max_length, data=None, dilation=1, num_deq=1, num_channels=1):
		self.in_pos = 0
		self.out_pos = 0
		self.num_deq = num_deq
		self.num_channels = num_channels
		self.dilation = dilation
		self.max_length = max_length
		self.data = data
		if data == None:
			self.data = torch.zeros(num_channels, max_length)

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
			t2 = self.data[:, self.out_pos % dilation:self.out_pos+1:dilation]
			t = torch.cat((t1, t2), 1)
		else:
			t = self.data[:, start:self.out_pos+1:dilation]

		self.out_pos = (self.out_pos + 1) % self.max_length
		return t

	def reset(self):
		self.data = torch.torch.zeros(self.num_channels, self.max_length)
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


class Model(nn.Module):
	def __init__(self,
				 num_time_samples=0,
				 num_channels=1,
				 num_classes=256,
				 num_kernel=2,
				 num_blocks=2,
				 num_layers=12,
				 num_hidden=128,
				 gpu_fraction=1.0):

		super(Model, self).__init__()

		self.num_time_samples = num_time_samples
		self.num_channels = num_channels
		self.num_classes = num_classes
		self.num_kernel = num_kernel
		self.num_blocks = num_blocks
		self.num_layers = num_layers
		self.num_hidden = num_hidden
		self.gpu_fraction = gpu_fraction

		main = nn.Sequential()

		scope = 0
		in_channels = 1
		out_channels = self.num_hidden

		# build model
		for b in range(num_blocks):
			num_additional = num_kernel - 1
			dilation = 1
			for i in range(num_layers):
				name = 'b{}-l{}.conv_dilation{}'.format(b, i, dilation)
				main.add_module(name, ConvDilated(in_channels,
												  out_channels,
												  kernel_size=num_kernel,
												  dilation=dilation))

				scope += num_additional
				num_additional *= 2
				dilation *= 2

				#print('b{}-l{}'.format(b, i))
				print('current scope: ', scope)

				#block_scope = block_scope * 2 + num_kernel - 1
				#print('block_scope: ', block_scope)
				in_channels = out_channels
			#scope += block_scope

		self.last_block_scope = 2**num_layers  # number of samples the last block generates
		scope = scope + self.last_block_scope
		print('scope: ', scope)

		main.add_module('final', Final(in_channels, num_classes))

		self.scope = scope # number of samples needed as input
		self.main = main

		#for parameter in self.parameters():
		#	ninit.constant(parameter, 1)

	def forward(self, input):
		# gpu_ids = None
		# if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
		# 	gpu_ids = range(self.ngpu)
		# x = nn.parallel.data_parallel(self.main, input, gpu_ids)
		x = self.main(input)
		return x

	def generate(self, num_generate, start_data=torch.FloatTensor([0])):
		"""
		:param start_data: torch tensor with length of at least one, used to start the generation process
		:param num_generate: number of samples that will be generated
		:return: torch tensor of size num_generated, containing the generated samples
		"""

		self.eval()
		#l = start_data.size(0)
		generated = Variable(start_data, volatile=True)

		num_pad = self.scope - generated.size(0)
		if num_pad > 0:
			generated = constant_pad_1d(generated, self.scope, pad_start=True)
			#ConstantPad1d(self.scope, pad_start=True).forward(generated)

		#if l > self.scope:
		#	start_data = start_data[l-start_data:l]

		for i in range(num_generate):
			input = generated[-self.scope:].view(1, 1, -1)
			o = self.forward(input)[-1, :].squeeze()
			max = torch.max(o, 0)[1].float()
			s = (max / self.num_classes) * 2. - 1 # new sample
			print(s.data[0])
			generated = torch.cat((generated, s), 0)

		return generated

	def fast_generate(self, num_generate, first_samples=torch.zeros((1))):
		self.eval()

		num_given_samples = first_samples.size()

		for module in self.modules():
			if hasattr(module, 'queue'):
				module.queue.reset()

		s = 0

		for i in range(num_given_samples):
			s = first_samples[i]
			for module in self.main.children():
				r = module.generate(s)

		generated = []

		for i in range(num_generate):
			for module in self.main.children():
				s = module.generate(s)
			#print(s[0])
			generated.append(s)

		return generated


class Optimizer:
	def __init__(self, model, learning_rate=0.001, train_hook=None, avg_length=20, stop_threshold=0.2):
		self.model = model
		self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		self.hook = train_hook
		self.avg_length = avg_length
		self.stop_threshold = stop_threshold

		#self.optimizer = optim.SGD(model.parameters(), lr=1000000., momentum=0.9)

	def train(self, data):
		self.model.train()  # set to train mode
		i = 0
		avg_loss = 0
		losses = []

		#indices = torch.randperm(data.data_length-self.model.scope) + self.model.scope
		indices = torch.randperm(data.data_length)
		#indices = [20000, 30000, 25000]
		while True:
			i += 1
			index = indices[i % indices.size(0)]
			self.optimizer.zero_grad()
			inputs, targets = data.get_minibatch([index])

			output = self.model(Variable(inputs))

			#print('step...')
			#labels = one_hot(targets, 256)
			loss = F.cross_entropy(output.squeeze(), Variable(targets))

			loss.backward()

			#print("parameters:")
			#for parameter in self.model.parameters():
			#	print(parameter.data)
			#	print(parameter.grad)

			self.optimizer.step()

			avg_loss += loss.data[0]

			if i % self.avg_length == 0:
				#print('output:', output[0].view(1, -1))
				#print('output max:', output.max(1)[1].view(1, -1))
				#print('targets:', targets.view(1, -1))
				avg_loss = avg_loss/20

				losses.append(avg_loss)
				if self.hook != None:
					self.hook(losses)

				if avg_loss < self.stop_threshold:
					print('save model')
					torch.save(self.model.state_dict(), 'last_trained_model')
					break

				avg_loss = 0

		# while not terminal:
		# 	i += 1
		#
		# 	self.optimizer.zero_grad()
		# 	output = self.model(_inputs)
		# 	loss = F.cross_entropy(output, _targets)
		# 	loss.backward()
		# 	self.optimizer.step()
		#
		# 	if loss < 1e-1:
		# 		terminal = True
		# 	losses.append(loss)
		# 	if i % 50 == 0:
		# 		plt.plot(losses)
		# 		plt.show()


class WavenetData:
	def __init__(self, path, input_length, target_length, num_classes):
		data = wavfile.read(path)[1][:, 0]

		max = np.max(data)

		#normalize
		max = np.maximum(np.max(data), -np.min(data))
		data = np.float32(data) / max

		bins = np.linspace(-1, 1, num_classes)
		# Quantize inputs.
		inputs = np.digitize(data[0:-1], bins, right=False) - 1
		inputs = bins[inputs]#[None, None, :]

		# Encode targets as ints.
		targets = (np.digitize(data[1::], bins, right=False) - 1)#[None, :]

		self.inputs = inputs
		self.targets = targets
		#print("inputs: ", inputs[10600:10900])
		#print("targets: ", targets[10600:10900])
		self.data_length = data.size
		self.input_length = input_length
		self.target_length = target_length
		self.num_classes = num_classes

	def get_minibatch(self, indices):
		# TODO: allow real minibatches
		#currently only one index possible
		this_input = []
		this_target = []
		for i in indices:
			if i < self.input_length:
				this_input = np.lib.pad(self.inputs[0:i],
										pad_width=(self.input_length-i, 0),
										mode='constant')
				this_input = this_input[None, None, :]
				#this_input = self.inputs[0:i][None, None, :]
			else:
				this_input = self.inputs[i-self.input_length:i][None, None, :]

			num_pad = self.target_length - i
			if num_pad > 0:
				pad = np.zeros(num_pad) + self.num_classes - 1
				this_target = np.concatenate((pad, self.targets[0:i]))
			else:
				this_target = self.targets[i-self.target_length:i]

		return torch.from_numpy(this_input).float(), torch.from_numpy(this_target).long()


