import torch
import torch.optim as optim
import time
from datetime import datetime
import librosa
import threading
import torch.nn.functional as F
from torch.autograd import Variable
from random import randint, shuffle

from wavenet_modules import *

def print_last_loss(losses):
	print("loss: ", losses[-1])

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
					date = str(datetime.now())
					print(date, ": snapshot saved to ", snapshot_file)
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
		mu_x = mu_law_econding(data, self.classes)
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
		For better performance, this can be called in a background thread.
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


