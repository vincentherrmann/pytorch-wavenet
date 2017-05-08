import torch
import torch.optim as optim
import time
import librosa
import threading
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.io import wavfile
from random import randint, shuffle
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


	def train(self,
			  data,
			  epochs=10,
			  epochs_per_snapshot=1,
			  file_path=None):
		self.model.train()  # set to train mode
		i = 0
		avg_loss = 0
		losses = []

		self.epoch = -1
		indices = self.new_epoch(data)

		m = self.mini_batch_size
		previous_loss = 1000

		while True:

			# check if epoch is completed
			if (i+1)*m > len(indices):
				if self.epoch >= epochs-1:
					print("training completed")
					break

				# new epoch
				i = 0
				avg_loss = 0
				indices = self.new_epoch(data)

				# make snapshots
				if self.epoch % epochs_per_snapshot == 0:
					if file_path != None:
						torch.save(self.model.state_dict(), file_path)
						print("snapshot saved to ", file_path)

			# get data
			self.optimizer.zero_grad()
			minibatch_indices = indices[i*m:(i+1)*m]
			inputs, targets = data.get_minibatch(minibatch_indices)
			targets = targets.view(targets.size(0) * targets.size(1))

			output = self.model(Variable(inputs))
			loss = F.cross_entropy(output.squeeze(), Variable(targets))
			loss.backward()

			loss = loss.data[0]
			if loss > previous_loss*3:
				print("unexpected high loss: ", loss)
				print("at minibatch ", minibatch_indices, " / ", data.data_length)

			self.optimizer.step()

			# train feedback
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



class WaveNetOptimizer:
	def __init__(self,
				 model,
				 mini_batch_size,
				 optimizer=optim.Adam,
				 learning_rate=0.001,
				 report_callback=print_last_loss,
				 report_length=20):

		self.model = model
		self.mini_batch_size = mini_batch_size
		self.optimizer = optimizer(params=self.model.parameters(),
								   lr=learning_rate)
		self.report_callback = report_callback
		self.report_length = report_length

	def new_training_order(self, index_count):
		l = self.model.output_length
		samples_count = index_count // l
		order = torch.randperm(samples_count-1)
		offset = randint(0, l-1)
		indices = [i*l + offset for i in order]
		return indices

	def train(self,
			  data,
			  epochs=10,
			  snapshot_interval=1800,
			  snapshot_file=None):

		self.model.train()  # set to train mode
		i = 0 # step index
		avg_loss = 0
		losses = []
		m = self.mini_batch_size
		previous_loss = 1000
		index_count = data.start_new_epoch()
		indices = self.new_training_order(index_count)
		epoch = 0
		tic = time.time()
		print("epoch ", epoch)

		# train loop
		while True:
			self.optimizer.zero_grad()

			# check if chunk is completed
			if (i + 1) * m > len(indices):
				index_count = data.use_new_chunk()

				# if epoch is finished
				if index_count == 0:
					# break if training is finished
					if epoch >= epochs-1:
						break
					index_count = data.start_new_epoch()
					epoch += 1
					print("epoch ", epoch)

				i = 0
				avg_loss = 0
				indices = self.new_training_order(index_count)

			# get data
			minibatch_indices = indices[i * m:(i + 1) * m]
			inputs, targets = data.get_wavenet_minibatch(minibatch_indices,
														 self.model.receptive_field,
														 self.model.output_length)
			targets = targets.view(targets.size(0) * targets.size(1))
			inputs = Variable(inputs)
			targets = Variable(targets)

			output = self.model(inputs)
			loss = F.cross_entropy(output.squeeze(), targets)
			loss.backward()

			loss = loss.data[0]
			# if loss > previous_loss * 3:
			# 	print("unexpected high loss: ", loss)
			# 	print("at minibatch ", minibatch_indices, " / ", data.data_length)

			self.optimizer.step()

			avg_loss += loss
			i += 1

			# train feedback
			if i % self.report_length == 0:
				avg_loss = avg_loss / self.report_length
				previous_loss = avg_loss
				losses.append(avg_loss)
				if self.report_callback != None:
					self.report_callback(losses)
				avg_loss = 0

			# snapshot
			toc = time.time()
			if toc - tic >= snapshot_interval:
				if snapshot_file != None:
					torch.save(self.model.state_dict(), snapshot_file)
					print("snapshot saved to ", snapshot_file)
				tic = toc





class AudioFileLoader:
	def __init__(self,
				 paths,
				 classes,
				 max_load_duration=3600,
				 dtype=torch.FloatTensor,
				 ltype=torch.LongTensor,
				 sampling_rate=11025):

		self.paths = paths
		self.sampling_rate = sampling_rate
		self.current_file = 0
		self.current_offset = 0
		self.classes = classes
		self.dtype = dtype
		self.ltype = ltype
		self.inputs = []
		self.targets = []
		self.loaded_data = []
		self.load_thread = threading.Thread(target=self.load_new_chunk, args=[])

		# calculate training data duration
		self.data_duration = 0
		for path in paths:
			d = librosa.get_duration(filename=path)
			self.data_duration += d
		print("total duration of training data: ", self.data_duration, " s")

		if self.data_duration < max_load_duration:
			self.max_load_duration = self.data_duration
		else:
			self.max_load_duration = max_load_duration

		#self.start_new_epoch()

	def quantize_data(self, data):
		# mu-law enconding
		mu_x = np.sign(data) * np.log(1 + self.classes*np.abs(data)) / np.log(self.classes + 1)
		# quantization
		bins = np.linspace(-1, 1, self.classes)
		quantized = np.digitize(mu_x, bins) - 1
		inputs = bins[quantized[0:-1]]
		targets = quantized[1::]
		return inputs, targets

	def start_new_epoch(self, shuffle_files=True):
		'''
		Start a new epoch. The order of the files can be shuffled.
		'''

		print("new epoch")
		if shuffle_files:
			shuffle(self.paths)
		self.current_file = 0
		self.current_offset = 0

		self.load_new_chunk(self.max_load_duration)
		return self.use_new_chunk()

	def use_new_chunk(self):
		'''
		Move the previously loaded data into a pytorch (cuda-)tensor and loads the next chunk in a background thread
		'''
		self.inputs = self.dtype(self.loaded_data[0])
		self.targets = self.ltype(self.loaded_data[1])

		while self.load_thread.is_alive():
			# wait...
			i=0

		# If there are no files left, return 0
		if self.current_file >= len(self.paths):
			#return self.start_new_epoch()
			return 0


		self.load_thread = threading.Thread(target=self.load_new_chunk, args=[self.max_load_duration])
		self.load_thread.start()
		return self.inputs.size(0)

	def load_new_chunk(self, duration):
		'''
		Load a new chunk of data with the specified duration (in seconds) from multiple disc files into memory.
		For better performance, this can be called from a background thread.
		'''
		print("load new chunk")
		loaded_duration = 0
		data = np.array(0)

		# loop though files until the specified duration is loaded
		while loaded_duration < duration:
			file = self.paths[self.current_file]
			remaining_duration = duration - loaded_duration
			print("data from ", file)
			#print("file number: ", self.current_file)
			#print("offset: ", self.current_offset)
			new_data, sr = librosa.load(self.paths[self.current_file],
										sr=self.sampling_rate,
										mono=True,
										offset=self.current_offset,
										duration=remaining_duration)
			new_loaded_duration = len(new_data) / self.sampling_rate
			data = np.append(data, new_data)
			loaded_duration += new_loaded_duration

			if new_loaded_duration < remaining_duration: # if this file was loaded completely
				# move to next file
				self.current_file += 1
				self.current_offset = 0
				# break if not enough data is available
				if self.current_file >= len(self.paths):
					break
			else:
				self.current_offset += new_loaded_duration

		data = self.quantize_data(data)
		self.loaded_data = data

	def get_wavenet_minibatch(self, indices, receptive_field, target_length):
		n = len(indices)
		input = self.dtype(n, 1, receptive_field).zero_()
		target = self.ltype(n, target_length).zero_()

		for i in range(n):
			index = indices[i] + 1

			sample_length = min(index, receptive_field)
			input[i, :, -sample_length:] = self.inputs[index-sample_length:index]

			sample_length = min(index, target_length)
			target[i, -sample_length:] = self.targets[index-sample_length:index]

		return input, target


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

