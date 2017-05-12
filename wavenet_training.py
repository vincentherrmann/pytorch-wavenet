import torch
import torch.optim as optim
import time
from datetime import datetime
import librosa
import threading
import torch.nn.functional as F
from torch.autograd import Variable
from random import randint, shuffle, uniform

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
        order = torch.randperm(samples_count - 1)
        offset = randint(0, l - 1)
        indices = [i * l + offset for i in order]
        return indices

    def train(self,
              data,
              epochs=10,
              snapshot_interval=1800,
              snapshot_file=None):

        self.model.train()  # set to train mode
        i = 0  # step index
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
                    if epoch >= epochs - 1:
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
                 receptive_field,
                 target_length,
                 max_load_duration=3600,
                 dtype=torch.FloatTensor,
                 ltype=torch.LongTensor,
                 sampling_rate=11025,
                 epoch_finished_callback=None):

        self.paths = paths
        self.sampling_rate = sampling_rate
        self.current_file = 0
        self.current_offset = 0
        self.classes = classes
        self.receptive_field = receptive_field
        self.target_length = target_length
        self.dtype = dtype
        self.ltype = ltype
        self.epoch_finished_callback = epoch_finished_callback

        # data that can be loaded in a background thread
        self.loaded_data = []
        self.load_thread = threading.Thread(target=self.load_new_chunk, args=[])

        # training data
        self.inputs = []
        self.targets = []
        self.training_indices = []
        self.training_index_count = 0
        self.current_training_index = 0

        # test data
        self.test_inputs = []
        self.test_targets = []
        self.test_index_count = 0
        self.test_positions

        # calculate training data duration
        self.data_duration = 0
        self.start_positions = [0]
        for path in paths:
            d = librosa.get_duration(filename=path)
            self.data_duration += d
            self.start_positions.append(self.data_duration)
        print("total duration of training data: ", self.data_duration, " s")

        if self.data_duration < max_load_duration:
            self.max_load_duration = self.data_duration
        else:
            self.max_load_duration = max_load_duration

            # self.start_new_epoch()

    def quantize_data(self, data):
        # mu-law enconding
        mu_x = mu_law_econding(data, self.classes)
        # quantization
        bins = np.linspace(-1, 1, self.classes)
        quantized = np.digitize(mu_x, bins) - 1
        inputs = bins[quantized[0:-1]]
        targets = quantized[1::]
        return inputs, targets


    def create_test_set(self, segments=32, minibatches_per_segment=8):
        '''
        Create test set from data that will be excluded from all training data
        '''

        self.test_inputs = []
        self.test_targets = []
        self.test_index_count = segments * minibatches_per_segment
        self.test_segment_duration = (self.target_length * minibatches_per_segment) / self.sampling_rate
        print("The test set has a total duration of ", segments*self.test_segment_duration, " s")

        available_segments = self.data_duration // self.test_segment_duration
        test_offset = uniform(0, self.test_segment_duration)
        self.test_positions = np.random.choice(available_segments, size=segments, replace=False) * self.test_segment_duration + test_offset
        additional_receptive_field = self.receptive_field - self.target_length # negative offset to accommodate for the receptive field

        for p in self.test_positions:
            d = self.load_segment(segment_position=p - additional_receptive_field,
                                  duration=self.test_segment_duration + additional_receptive_field + 1) # + 1 for the targets
            i, t = self.quantize_data(d)
            self.test_inputs.append(i)
            self.test_targets.append(t)


    def get_test_minibatches(self, minibatch_size=8):
        minibatches = []
        current_index = 0
        #while current_index < self.test_index_count:


    def start_new_epoch(self, segments_per_chunk=16, minibatches_per_segment=32):
        self.segments_per_chunk = segments_per_chunk
        self.training_index_count = segments_per_chunk * minibatches_per_segment
        self.training_segment_duration = (self.target_length * minibatches_per_segment) / self.sampling_rate

        training_offset = uniform(0, self.training_segment_duration)
        available_segments = self.data_duration // self.training_segment_duration

        self.segment_positions = np.random.permutation(available_segments) * self.training_segment_duration + training_offset
        self.chunk_position = 0


    def load_new_chunk(self):
        self.loaded_data = []
        current_chunk_position = self.chunk_position

        while len(self.loaded_data) < self.segments_per_chunk:
            segment_position = self.segment_positions[current_chunk_position]

            # check if this segment overlaps with any test segment,
            # if yes, then block it
            segment_is_blocked = False
            for test_position in self.test_positions:
                train_seg_end = segment_position + self.training_segment_duration
                test_seg_end = test_position + self.test_segment_duration
                if train_seg_end > test_position & segment_position < test_seg_end:
                    segment_is_blocked = True
                    break

            current_chunk_position += 1

            if current_chunk_position > len(self.segment_positions):
                if self.epoch_finished_callback != None:
                    self.new_epoch_callback()
                    break
                else:
                    break

            if segment_is_blocked:
                continue

            new_data = self.load_segment(segment_position, self.training_segment_duration)
            i, t = self.quantize_data(new_data)
            self.loaded_data.append((i, t))

        self.chunk_position = current_chunk_position


    def use_new_chunk(self):
        # wait for loading to finish
        self.load_thread.join()
        self.sample_indices = np.random.permutation(self.training_index_count)
        self.current_training_index = 0
        self.inputs = []
        self.targets = []

        for inputs, targets in self.loaded_data:
            self.inputs.append(self.dtype(inputs))
            self.targets.append(self.ltype(inputs))

        self.load_thread = threading.Thread(target=self.load_new_chunk)
        self.load_thread.start()


    def get_minibatch(self, minibatch_size):
        input = self.dtype(minibatch_size, 1, self.receptive_field).zero_()
        target = self.ltype(minibatch_size, self.target_length).zero_()

        for i in range(minibatch_size):
            index = self.sample_indices[self.current_training_index]
            segment = index // self.segments_per_chunk
            position = (index % self.segments_per_chunk) * self.target_length + self.rec + 1

            sample_length = min(position, self.receptive_field)
            input[i, :, -sample_length] = self.inputs[segment][index - sample_length:index]

            sample_length = min(position, self.target_length)
            target[i, :, -sample_length] = self.inputs[segment][index - sample_length:index]

            self.current_training_index += 1
            if self.current_training_index > len(self.training_index_count):
                self.use_new_chunk()

        return input, target



    def get_wavenet_minibatch(self, indices, receptive_field, target_length):
        n = len(indices)
        input = self.dtype(n, 1, receptive_field).zero_()
        target = self.ltype(n, target_length).zero_()

        for i in range(n):
            index = indices[i] + 1

            sample_length = min(index, receptive_field)
            input[i, :, -sample_length:] = self.inputs[index - sample_length:index]

            sample_length = min(index, target_length)
            target[i, -sample_length:] = self.targets[index - sample_length:index]

        return input, target




    def load_segment(self, segment_position, duration):
        # find the right file
        file_index = 0
        while self.start_positions[file_index+1] < segment_position:
            file_index += 1
            if file_index+1 > len(self.start_positions):
                print("position ", segment_position, "is to not available")
                return np.array()
        file_path = self.paths[file_index]

        # load from file
        offset = segment_position - self.start_positions[file_index]
        new_data, sr = librosa.load(self.paths[self.current_file],
                                    sr=self.sampling_rate,
                                    mono=True,
                                    offset=self.offset,
                                    duration=duration)

        # if the file was not long enough, recursively call this function on the next file to get the remaining duration
        new_loaded_duration = len(new_data) / self.sampling_rate
        if new_loaded_duration < duration:
            new_position = self.start_positions[file_index+1]
            new_duration = duration - new_loaded_duration
            additional_data = self.load_segment(new_position, new_duration)
            new_data = np.append(new_data, additional_data)

        return new_data




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
            i = 0

        # If there are no files left, return 0
        if self.current_file >= len(self.paths):
            # return self.start_new_epoch()
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
            # print("file number: ", self.current_file)
            # print("offset: ", self.current_offset)
            new_data, sr = librosa.load(self.paths[self.current_file],
                                        sr=self.sampling_rate,
                                        mono=True,
                                        offset=self.current_offset,
                                        duration=remaining_duration)
            new_loaded_duration = len(new_data) / self.sampling_rate
            data = np.append(data, new_data)
            loaded_duration += new_loaded_duration

            if new_loaded_duration < remaining_duration:  # if this file was loaded completely
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

