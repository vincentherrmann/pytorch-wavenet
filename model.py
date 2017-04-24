import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.io import wavfile
from random import randint
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
				residual = True #in_channels == out_channels
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


class WaveNetModel(nn.Module):
	def __init__(self,
				 num_layers,
				 num_blocks,
				 num_classes,
				 kernel_size=2,
				 hidden_channels=128,
				 sampled_generation=False):

		super(WaveNetModel, self).__init__()

		self.num_classes = num_classes
		self.sampled_generation = sampled_generation

		main = nn.Sequential()

		scope = 0
		in_channels = 1
		out_channels = hidden_channels
		init_dilation = 1

		# build model
		for b in range(num_blocks):
			additional_scope = kernel_size-1
			new_dilation = 1
			for i in range(num_layers):
				name = "b{}-l{}.wavenet_layer-d{}".format(b, i, new_dilation)
				main.add_module(name, WaveNetLayer(in_channels,
												   out_channels,
												   kernel_size=kernel_size,
												   dilation=new_dilation,
												   init_dilation=init_dilation,
												   residual_connection=True))
				scope += additional_scope
				additional_scope *= 2
				init_dilation = new_dilation
				new_dilation *= 2
				in_channels = out_channels

		self.last_block_scope = 2**num_layers
		scope = scope + self.last_block_scope

		main.add_module('final', WaveNetFinalLayer(in_channels,
												   num_classes=num_classes,
												   out_length=self.last_block_scope,
												   init_dilation=init_dilation))

		self.scope = scope
		self.main = main

	def forward(self, input):
		x = self.main(input)
		return x

	def generate(self, num_generate, start_data=torch.zeros((1))):
		self.eval()
		generated = Variable(start_data, volatile=True)

		num_pad = self.scope - generated.size(0)
		if num_pad > 0:
			generated = constant_pad_1d(generated, self.scope, pad_start=True)
			print("pad zero")

		for i in range(num_generate):
			input = generated[-self.scope:].view(1, 1, -1)
			o = self(input)[-1, :].squeeze()

			if self.sampled_generation:
				soft_o = F.softmax(o)
				np_o = soft_o.data.numpy()
				s = np.random.choice(self.num_classes, p=np_o)
				s = Variable(torch.FloatTensor([s]))
				s = (s / self.num_classes) * 2. - 1
			else:
				max = torch.max(o, 0)[1].float()
				s = (max / self.num_classes) * 2. - 1  # new sample

			generated = torch.cat((generated, s), 0)

		return generated.data.tolist()

	def fast_generate(self,
					  num_generate,
					  first_samples=torch.zeros((1)),
					  progress_callback=None):
		self.eval()

		num_given_samples = first_samples.size(0)
		total_samples = num_given_samples + num_generate
		progress_dist = total_samples // 100

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

			if i % progress_dist == 0:
				if progress_callback != None:
					progress_callback(i, total_samples)

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
				max = torch.max(s, 0)[1][0] #.data[0]
				s = (max / self.num_classes) * 2. - 1 # new sample

			if (i+num_given_samples) % progress_dist == 0:
				if progress_callback != None:
					progress_callback(i+num_given_samples, total_samples)

			generated.append(s)

		return generated


def print_last_loss(losses):
	print("loss: ", losses[-1])


class Optimizer:
	def __init__(self, model, mini_batch_size, stop_threshold=0, learning_rate=0.001, train_hook=print_last_loss, avg_length=20):
		self.model = model
		self.optimizer = optim.Adam(model.parameters(),
									lr=learning_rate)
		self.hook = train_hook
		self.avg_length = avg_length
		self.stop_threshold = stop_threshold
		self.mini_batch_size = mini_batch_size
		self.epoch = 0


	def train(self, data, epochs=10):
		self.model.train()  # set to train mode
		i = 0
		avg_loss = 0
		epoch = -1
		losses = []

		self.epoch = -1
		indices = self.new_epoch(data)

		m = self.mini_batch_size
		previous_loss = 1000

		while True:
			if (i+1)*m > len(indices):
				if epoch >= epochs-1:
					print("training completed")
					break
				i = 0
				avg_loss = 0
				indices = self.new_epoch(data)
				epoch += 1

			self.optimizer.zero_grad()
			minibatch_indices = indices[i*m:(i+1)*m]
			inputs, targets = data.get_minibatch(minibatch_indices)

			output = self.model(Variable(inputs))

			targets = targets.view(targets.size(0)*targets.size(1))
			loss = F.cross_entropy(output.squeeze(), Variable(targets))
			loss.backward()

			loss = loss.data[0]
			if loss > previous_loss*5:
				print("unexpect high loss: ", loss)
				print("at minibatch ", minibatch_indices)

			self.optimizer.step()

			avg_loss += loss
			i += 1
			if i % self.avg_length == 0:
				avg_loss = avg_loss / self.avg_length
				previous_loss = avg_loss
				losses.append(avg_loss)
				if self.hook != None:
					self.hook(losses)

				if avg_loss < self.stop_threshold:
					print('training completed with loss', avg_loss)
					break
				avg_loss = 0


	def new_epoch(self, data):
		self.epoch += 1

		l = data.target_length
		samples_count = data.data_length // l
		indices = torch.randperm(samples_count-1)
		offset = randint(0, l-1)
		indices = [i*l + offset for i in indices]

		print("epoch ", self.epoch)
		return indices





class WaveNetData:
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
		targets = np.digitize(data[1::], bins, right=False) - 1 #[None, :]

		self.inputs = inputs
		self.targets = targets
		self.data_length = data.size
		self.input_length = input_length
		self.target_length = target_length
		self.num_classes = num_classes

	def get_minibatch(self, indices):
		l = len(indices)
		this_input = np.zeros((l, 1, self.input_length))
		this_target = np.zeros((l, self.target_length))

		for i in range(l):
			index = indices[i]+1
			sample_length = min(index, self.input_length)
			this_input[i, :, -sample_length:] = self.inputs[index-sample_length:index]
			sample_length = min(index, self.target_length)
			this_target[i, -sample_length:] = self.targets[index-sample_length:index]

		# for i in indices:
		# 	if i < self.input_length:
		# 		this_input = np.lib.pad(self.inputs[0:i],
		# 								pad_width=(self.input_length-i, 0),
		# 								mode='constant')
		# 		this_input = this_input[None, None, :]
		# 	else:
		# 		this_input = self.inputs[i-self.input_length:i][None, None, :]
		#
		# 	num_pad = self.target_length - i
		# 	if num_pad > 0:
		# 		pad = np.zeros(num_pad) + self.num_classes - 1
		# 		this_target = np.concatenate((pad, self.targets[0:i]))
		# 	else:
		# 		this_target = self.targets[i-self.target_length:i]

		return torch.from_numpy(this_input).float(), torch.from_numpy(this_target).long()


