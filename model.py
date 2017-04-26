import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.io import wavfile
from random import randint
from wavenet_modules import *


class WaveNetModel(nn.Module):

	def __init__(self,
				 num_layers,
				 num_blocks,
				 num_classes,
				 kernel_size=2,
				 hidden_channels=128):

		super(WaveNetModel, self).__init__()

		self.num_classes = num_classes

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

		self.last_block_scope = 2**(num_layers-1)
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
					  progress_callback=None,
					  sampled_generation=False):
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

			if sampled_generation:
				soft_o = F.softmax(Variable(s))
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

	def parameter_count(self):
		par = list(self.parameters())
		s = sum([np.prod(list(d.size())) for d in par])
		return s


def print_last_loss(losses):
	print("loss: ", losses[-1])


class Optimizer:
	def __init__(self, model,
				 mini_batch_size,
				 stop_threshold=0,
				 learning_rate=0.001,
				 train_hook=print_last_loss,
				 avg_length=20):
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
			if loss > previous_loss*3:
				print("unexpected high loss: ", loss)
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
	def __init__(self, path, input_length, target_length, num_classes, cuda=False):
		data = wavfile.read(path)[1][:, 0]

		#normalize
		max = np.maximum(np.max(data), -np.min(data))
		data = np.float32(data) / max

		# Quantize inputs.
		bins = np.linspace(-1, 1, num_classes)
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
		self.cuda = cuda

	def get_minibatch(self, indices):
		"""
		:param indices: A list of indices that mark the last sample of each example in the minibatch
		:return: (examples, target) - A tuple of torch tensors, the minibatch and the corresponding labels
		"""
		l = len(indices)
		this_input = np.zeros((l, 1, self.input_length))
		this_target = np.zeros((l, self.target_length))

		for i in range(l):
			index = indices[i]+1
			sample_length = min(index, self.input_length)
			this_input[i, :, -sample_length:] = self.inputs[index-sample_length:index]
			sample_length = min(index, self.target_length)
			this_target[i, -sample_length:] = self.targets[index-sample_length:index]

		b = torch.from_numpy(this_input).float()
		t = torch.from_numpy(this_target).long()

		if self.cuda:
			b = b.cuda()
			t = t.cuda()

		return (b, t)

