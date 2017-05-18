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


def print_last_loss(opt):
    print("loss: ", opt.losses[-1])


def print_last_test_result(opt):
    print("test loss: ", opt.test_results[-1])


class WaveNetOptimizer:
    def __init__(self,
                 model,
                 data,
                 test_segments=0,
                 examples_per_test_segment=8,
                 optimizer=optim.Adam,
                 report_callback=print_last_loss,
                 report_interval=8,
                 test_report_callback=print_last_test_result,
                 test_interval=64,
                 snapshot_interval=256,
                 snapshot_file=None):

        self.model = model
        self.data = data
        self.data.epoch_finished_callback = self.new_epoch
        self.optimizer = optimizer(params=self.model.parameters())

        if test_segments > 0:
            self.data.create_test_set(segments=test_segments,
                                      examples_per_segment=examples_per_test_segment)

        self.report_callback = report_callback
        self.report_interval = report_interval
        self.test_report_callback = test_report_callback
        self.test_interval = test_interval
        self.snapshot_interval = snapshot_interval
        self.snapshot_file = snapshot_file

        self.losses = []
        self.loss_positions = []
        self.test_results = []
        self.test_result_positions = []
        self.current_epoch = 0
        self.epochs = 1
        self.segments_per_chunk = 16
        self.examples_per_segment = 32

    def new_epoch(self):
        '''
        Start a new epoch or end training
        '''
        self.current_epoch += 1
        if self.current_epoch >= self.epochs:
            print("training finished")
            return

        print("epoch ", self.current_epoch)
        self.data.start_new_epoch(segments_per_chunk=self.segments_per_chunk,
                                  examples_per_segment=self.examples_per_segment)

    def test_model(self, position, test_m=16):
        '''
        Run model on test set and report the result

        :param test_m: number of examples from the test set in one minibatch
        '''
        self.model.eval()
        avg_loss = 0
        i = 0

        while i < self.data.test_index_count:
            inputs = self.data.test_inputs[i:(i+test_m), :, :]
            inputs = Variable(inputs, volatile=True)
            targets = self.data.test_targets[i:(i+test_m), :]
            targets = targets.view(targets.size(0) * targets.size(1))
            targets = Variable(targets, volatile=True)
            output = self.model(inputs)
            loss = F.cross_entropy(output.squeeze(), targets).data[0]
            avg_loss += loss
            i += test_m

        avg_loss = avg_loss * test_m / self.data.test_index_count
        self.test_results.append(avg_loss)
        self.test_result_positions.append(position)
        if self.test_report_callback is not None:
            self.test_report_callback(self)
        #print("test loss: ", avg_loss)

        self.model.train()

    def train(self,
              learning_rate=0.001,
              minibatch_size=8,
              epochs=100,
              segments_per_chunk=16,
              examples_per_segment=32):

        '''
        Train a Wavenet model

        :param learning_rate: Learning rate of the optimizer
        :param minibatch_size: Number of examples in one minibatch
        :param epochs: Number of training epochs
        :param segments_per_chunk: Number of segments from the training data that are simultaneously loaded into memory
        :param examples_per_segment: The number of examples each of these segments contains
        '''

        self.optimizer.lr = learning_rate
        self.epochs = epochs
        self.segments_per_chunk = segments_per_chunk
        self.examples_per_segment = examples_per_segment

        self.model.train()  # set to train mode
        i = 0  # step index
        avg_loss = 0
        losses = []
        previous_loss = 1000
        self.current_epoch = -1

        self.new_epoch()
        self.data.load_new_chunk()
        self.data.use_new_chunk()

        # train loop
        while True:
            self.optimizer.zero_grad()

            # get data
            inputs, targets = self.data.get_minibatch(minibatch_size)
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
            if i % self.report_interval == 0:
                avg_loss /= self.report_interval
                previous_loss = avg_loss

                self.losses.append(avg_loss)
                self.loss_positions.append(i)

                if self.report_callback != None:
                    self.report_callback(self)
                avg_loss = 0

            # run on test set
            if i % self.test_interval == 0:
                self.test_model(i)

            # snapshot
            if i % self.snapshot_interval == 0:
                if self.snapshot_file != None:
                    torch.save(self.model.state_dict(), self.snapshot_file)
                    date = str(datetime.now())
                    print(date, ": snapshot saved to ", self.snapshot_file)


class AudioFileLoader:
    def __init__(self,
                 paths,
                 classes,
                 receptive_field,
                 target_length,
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
        self.load_thread = None #threading.Thread(target=self.load_new_chunk, args=[])

        # training data
        self.inputs = []
        self.targets = []
        self.training_indices = []
        self.training_index_count = 0
        self.training_segment_duration = 0.
        self.current_training_index = 0

        self.segments_per_chunk = 0
        self.examples_per_segment = 0
        self.segment_positions = np.array(0)
        self.chunk_position = 0
        self.additional_receptive_field = (self.receptive_field - self.target_length + 1) / self.sampling_rate # negative offset to accommodate for the receptive field

        # test data
        self.test_inputs = np.array(0)
        self.test_targets = np.array(0)
        self.test_index_count = 0
        self.test_positions = []
        self.test_segment_duration = 0.

        # calculate training data duration
        self.data_duration = 0
        self.start_positions = [0]
        for path in paths:
            d = librosa.get_duration(filename=path)
            self.data_duration += d
            self.start_positions.append(self.data_duration)
        print("total duration of training data: ", self.data_duration, " s")

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

    def create_test_set(self, segments=32, examples_per_segment=8):
        '''
        Create test set from data that will be excluded from all training data
        '''

        self.test_index_count = segments * examples_per_segment
        self.test_inputs = self.dtype(self.test_index_count, 1, self.receptive_field).zero_()
        self.test_targets = self.ltype(self.test_index_count, self.target_length).zero_()

        self.test_segment_duration = (self.target_length * examples_per_segment) / self.sampling_rate
        print("The test set has a total duration of ", segments*self.test_segment_duration, " s")

        available_segments = int(self.data_duration // self.test_segment_duration) - 1 # number of segments that can be chosen from
        test_offset = uniform(0, self.test_segment_duration) # some random offset
        positions = np.random.choice(available_segments, size=segments, replace=False)
        self.test_positions = positions * self.test_segment_duration + test_offset

        duration = self.test_segment_duration + self.additional_receptive_field

        for s in range(segments):
            position = self.test_positions[s] - self.additional_receptive_field
            d = self.load_segment(segment_position=position,
                                  duration=duration)
            i, t = self.quantize_data(d)
            i = self.dtype(i)

            t_temp = torch.from_numpy(t)
            if self.ltype.is_cuda:
                t_temp = t_temp.cuda()
            t = self.ltype(t_temp)

            for m in range(examples_per_segment):
                example_index = s * examples_per_segment + m
                position = m*self.target_length + self.receptive_field
                if position > i.size(0):
                    print("index ", position, " is not avialable in a tensor of size ", i.size(0))
                self.test_inputs[example_index, :, :] = i[position - self.receptive_field:position]
                self.test_targets[example_index, :] = t[position - self.target_length:position]

    def start_new_epoch(self, segments_per_chunk=16, examples_per_segment=32):
        # wait for loading to finish
        # if self.load_thread != None:
        #     if self.load_thread.is_alive():
        #         self.load_thread.join()

        #print("\n start new epoch")

        self.segments_per_chunk = segments_per_chunk
        self.examples_per_segment = examples_per_segment
        self.training_index_count = segments_per_chunk * examples_per_segment
        self.training_segment_duration = (self.target_length * examples_per_segment) / self.sampling_rate

        training_offset = uniform(0, self.training_segment_duration)
        available_segments = int(self.data_duration // self.training_segment_duration) - 1

        if(available_segments < segments_per_chunk):
            print("There are not enough segments available in the training set to produce one chunk")

        self.segment_positions = np.random.permutation(available_segments) * self.training_segment_duration + training_offset
        self.chunk_position = 0
        #print("with positions: ", self.segment_positions)

        #self.load_new_chunk()
        #self.use_new_chunk()

    def load_new_chunk(self):
        tic = time.time()
        #print("load new chunk with start segment index ", self.chunk_position)
        self.loaded_data = []
        current_chunk_position = self.chunk_position

        while len(self.loaded_data) < self.segments_per_chunk:
            if current_chunk_position >= len(self.segment_positions):
                #print("epoch finished")
                if self.epoch_finished_callback != None:
                    self.epoch_finished_callback()
                current_chunk_position = self.chunk_position
                #break

            segment_position = self.segment_positions[current_chunk_position]

            # check if this segment overlaps with any test segment,
            # if yes, then block it
            segment_is_blocked = False
            for test_position in self.test_positions:
                train_seg_end = segment_position + self.training_segment_duration
                test_seg_end = test_position + self.test_segment_duration
                if (train_seg_end > test_position) & (segment_position < test_seg_end):
                    #print("block segment at position ", test_position)
                    segment_is_blocked = True
                    break

            current_chunk_position += 1

            if segment_is_blocked:
                continue

            new_data = self.load_segment(segment_position, self.training_segment_duration + self.additional_receptive_field)
            i, t = self.quantize_data(new_data)
            self.loaded_data.append((i, t))

        self.training_index_count = len(self.loaded_data) * self.examples_per_segment
        self.chunk_position = current_chunk_position
        #print("there are ", len(self.loaded_data), " segments in the newly loaded chunk")
        toc = time.time()
        if toc-tic > 60:
            print("loading this chunk took ", toc-tic, " seconds")


    def use_new_chunk(self):
        #print("use loaded chunk with ", len(self.loaded_data), "segments")

        # wait for loading to finish
        if self.load_thread != None:
            if self.load_thread.is_alive():
                print("Loading the data is slowing the training process down. Maybe you should use less segments per chunk or uncompressed audio files.")
                self.load_thread.join()

        if len(self.loaded_data) == 0:
            print("no data loaded?!")

        self.sample_indices = np.random.permutation(self.training_index_count)
        #print("last training index count: ", self.training_index_count)
        self.current_training_index = 0

        if len(self.inputs) >= self.segments_per_chunk:
            self.inputs = []
            self.targets = []

        for inputs, targets in self.loaded_data:
            self.inputs.append(self.dtype(inputs))
            t_temp = torch.from_numpy(targets)
            if self.ltype.is_cuda:
                t_temp = t_temp.cuda()
            self.targets.append(self.ltype(t_temp))

        #self.load_new_chunk()
        self.load_thread = threading.Thread(target=self.load_new_chunk)
        self.load_thread.start()

    def get_minibatch(self, minibatch_size):
        #print("    load minibatch")
        input = self.dtype(minibatch_size, 1, self.receptive_field).zero_()
        target = self.ltype(minibatch_size, self.target_length).zero_()

        if self.training_index_count < minibatch_size:
            print("not enough data for one minibatch in chunk. You should probably load bigger chunks into memory.")

        for i in range(minibatch_size):
            index = self.sample_indices[self.current_training_index]
            segment = index // self.examples_per_segment
            position = (index % self.examples_per_segment) * self.target_length + self.receptive_field

            if position > self.inputs[segment].size(0):
                print("index ", position, " is not avialable in a tensor of size ", self.inputs[segment].size(0))

            sample_length = min(position, self.receptive_field)
            input[i, :, -sample_length:] = self.inputs[segment][(position - sample_length):position]

            sample_length = min(position, self.target_length)
            target[i, -sample_length:] = self.targets[segment][(position - sample_length):position]

            self.current_training_index += 1
            if self.current_training_index >= self.training_index_count:
                #print("use new chunk")
                self.use_new_chunk()

        return input, target

    def load_segment(self, segment_position, duration):
        # find the right file
        file_index = 0
        while self.start_positions[file_index+1] <= segment_position:
            file_index += 1
            if file_index+1 >= len(self.start_positions):
                print("position ", segment_position, "is not available")
                return np.array(0)
        file_path = self.paths[file_index]

        # load from file
        offset = segment_position - self.start_positions[file_index]
        new_data, sr = librosa.load(path=file_path,
                                    sr=self.sampling_rate,
                                    mono=True,
                                    offset=offset,
                                    duration=duration)

        # if the file was not long enough, recursively call this function on the next file to get the remaining duration
        new_loaded_duration = len(new_data) / self.sampling_rate
        if new_loaded_duration < duration-0.00001:
            new_position = self.start_positions[file_index+1]
            new_duration = duration - new_loaded_duration
            additional_data = self.load_segment(new_position, new_duration)
            new_data = np.append(new_data, additional_data)

        return new_data