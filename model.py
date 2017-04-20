import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.io import wavfile
from wavenet_modules import *


class Model(nn.Module):
	def __init__(self,
				 num_time_samples=0,
				 num_channels=1,
				 num_classes=256,
				 num_kernel=2,
				 num_blocks=2,
				 num_layers=12,
				 num_hidden=128,
				 gpu_fraction=1.0,
				 sampled_generation=False):

		super(Model, self).__init__()

		self.num_time_samples = num_time_samples
		self.num_channels = num_channels
		self.num_classes = num_classes
		self.num_kernel = num_kernel
		self.num_blocks = num_blocks
		self.num_layers = num_layers
		self.num_hidden = num_hidden
		self.gpu_fraction = gpu_fraction
		self.sampled_generation = sampled_generation

		main = nn.Sequential()

		scope = 0
		in_channels = 1
		out_channels = self.num_hidden

		# build model
		for b in range(num_blocks):
			num_additional = num_kernel - 1
			dilation = 1
			for i in range(num_layers):
				residual = in_channels == out_channels
				name = 'b{}-l{}.conv_dilation{}'.format(b, i, dilation)
				main.add_module(name, ConvDilated(in_channels,
												  out_channels,
												  kernel_size=num_kernel,
												  dilation=dilation,
												  residual_connection=residual))

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

		main.add_module('final', Final(in_channels, num_classes, self.last_block_scope))

		self.scope = scope # number of samples needed as input
		self.main = main

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
		generated = Variable(start_data, volatile=True)

		num_pad = self.scope - generated.size(0)
		if num_pad > 0:
			generated = constant_pad_1d(generated, self.scope, pad_start=True)
			print("pad zero")
			#ConstantPad1d(self.scope, pad_start=True).forward(generated)

		for i in range(num_generate):
			input = generated[-self.scope:].view(1, 1, -1)
			o = self.forward(input)[-1, :].squeeze()

			if self.sampled_generation:
				soft_o = F.softmax(o)
				np_o = soft_o.data.numpy()
				s = np.random.choice(self.num_classes, p=np_o)
				s = Variable(torch.FloatTensor([s]))
				s = (s / self.num_classes) * 2. - 1
			else:
				max = torch.max(o, 0)[1].float()
				s = (max / self.num_classes) * 2. - 1 # new sample

			generated = torch.cat((generated, s), 0)

		return generated


	def compare_generate(self, num_generate, first_samples=torch.zeros((1))):
		self.eval()
		num_given_samples = first_samples.size(0)

		fast_support_generated = []
		s = first_samples[0]
		# for i in range(num_given_samples-1):
		# 	r = s
		# 	s = first_samples[i+1]
		# 	for module in self.main.children():
		# 		r = module.generate(r)
		# 	fast_support_generated.append(r)

		conv_generated = Variable(first_samples, volatile=True)
		fast_generated = []
		fast_input = s

		for i in range(num_generate):
			print("sample number: ", i)
			# generate conventional
			conv_input = conv_generated[-self.scope:].view(1, 1, -1)
			conv_output = self.forward(conv_input)[-1, :].squeeze()
			conv_max = torch.max(conv_output, 0)[1].float()
			conv_new = (conv_max / self.num_classes) * 2. - 1
			conv_generated = torch.cat((conv_generated, conv_new), 0)

			# generate fast
			for module in self.main.children():
				s = module.generate(s)
			fast_output = s
			fast_max = torch.max(fast_output, 0)[1].data[0]
			fast_new = (fast_max / self.num_classes) * 2. - 1
			s = fast_new
			fast_generated.append(fast_new)

			if torch.equal(conv_output.data, fast_output.data) == False:
				print("conv_output: ", conv_output.data)
				print("fast_output: ", fast_output.data)
				print("difference in output at i=", i)

			if conv_new.data[0] != fast_new:
				print("difference in generated sample at i=", i)

		return [conv_generated, fast_generated]


	def fast_generate(self, num_generate, first_samples=torch.zeros((1))):
		self.eval()

		num_given_samples = first_samples.size(0)

		# reset queues
		for module in self.modules():
			if hasattr(module, 'queue'):
				module.queue.reset()

		s = first_samples[0]

		# create samples with the support from the first_samples
		support_generated = []
		for i in range(num_given_samples-1):
			# replace generated sample by provided sample
			r = s
			s = first_samples[i+1]
			for module in self.main.children():
				r = module.generate(r)
			support_generated.append(r)

		# autoregressive sample generation
		generated = []
		for i in range(num_generate):
			for module in self.main.children():
				s = module.generate(s)

			if self.sampled_generation:
				soft_o = F.softmax(s)
				np_o = soft_o.data.numpy()
				s = np.random.choice(self.num_classes, p=np_o)
				s = (s / self.num_classes) * 2. - 1
			else:
				max = torch.max(s, 0)[1].data[0]
				s = (max / self.num_classes) * 2. - 1 # new sample

			generated.append(s)

		return [generated, support_generated]


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
				avg_loss = avg_loss / self.avg_length

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


