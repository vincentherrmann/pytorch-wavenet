import torch
import torch.optim as optim
import time
from datetime import datetime
import librosa
import threading
import torch.nn.functional as F
from torch.autograd import Variable
from random import randint, shuffle, uniform
from logger import Logger

from wavenet_modules import *


def print_last_loss(opt):
    print("loss: ", opt.losses[-1])


def print_last_validation_result(opt):
    print("validation loss: ", opt.validation_results[-1])


class WaveNetOptimizer:
    def __init__(self,
                 model,
                 data,
                 validation_segments=0,
                 examples_per_validation_segment=8,
                 optimizer=optim.Adam,
                 report_callback=print_last_loss,
                 report_interval=8,
                 validation_report_callback=print_last_validation_result,
                 logging_interval=64,
                 validation_interval=64,
                 snapshot_interval=256,
                 snapshot_file=None,
                 segments_per_chunk=16,
                 examples_per_segment=32):

        self.model = model
        self.data = data
        self.data.epoch_finished_callback = self.new_epoch
        self.learning_rate = 0.001
        self.optimizer_type = optimizer
        self.optimizer = optimizer(params=self.model.parameters(), lr=self.learning_rate)

        if validation_segments > 0:
            self.data.create_validation_set(segments=validation_segments,
                                            examples_per_segment=examples_per_validation_segment)

        self.report_callback = report_callback
        self.report_interval = report_interval
        self.validation_report_callback = validation_report_callback
        self.validation_interval = validation_interval
        self.logging_interval = logging_interval
        self.snapshot_interval = snapshot_interval
        self.snapshot_file = snapshot_file
        self.logger = Logger('./logs')


        self.i = 0 # current step
        self.losses = []
        self.step_times = []
        self.loss_positions = []
        self.validation_results = []
        self.validation_result_positions = []
        self.avg_loss = 0
        self.avg_time = 0
        self.current_epoch = -1
        self.epochs = 1
        self.segments_per_chunk = segments_per_chunk
        self.examples_per_segment = examples_per_segment

        self.new_epoch()
        self.data.load_new_chunk()
        self.data.use_new_chunk()

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

    def validate_model(self, position, validation_m=16):
        '''
        Run model on validation set and report the result

        :param validation_m: number of examples from the validation set in one minibatch
        '''
        self.model.eval()
        avg_loss = 0
        i = 0

        while i < self.data.validation_index_count:
            inputs = self.data.validation_inputs[i:(i + validation_m), :, :]
            inputs = Variable(inputs, volatile=True)
            targets = self.data.validation_targets[i:(i + validation_m), :]
            targets = targets.view(targets.size(0) * targets.size(1))
            targets = Variable(targets, volatile=True)
            output = self.model(inputs)
            loss = F.cross_entropy(output.squeeze(), targets).data[0]
            avg_loss += loss
            i += validation_m

        avg_loss = avg_loss * validation_m / self.data.validation_index_count
        self.validation_results.append(avg_loss)
        self.validation_result_positions.append(position)
        if self.validation_report_callback is not None:
            self.validation_report_callback(self)

        self.model.train()

    def log_to_tensor_board(self):
        # TensorBoard logging

        # loss
        self.logger.scalar_summary("loss", self.avg_loss, self.i)

        # parameter histograms
        for tag, value, in self.model.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary(tag, value.data.cpu().numpy(), self.i)
            if value.grad is not None:
                self.logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), self.i)

        # normalized cross correlation
        for tag, module in self.model.named_modules():
            tag = tag.replace('.', '/')
            if type(module) is Conv1dExtendable:
                ncc = module.normalized_cross_correlation()
                self.logger.histo_summary(tag + '/ncc', ncc.data.cpu().numpy(), self.i)

    def log_normalized_cross_correlation(self):
        print("cross correlations")
        for name, module in self.model.named_modules():
            if type(module) is Conv1dExtendable:
                ncc = module.normalized_cross_correlation()
                print(ncc)

    def split_important_features(self, threshold):
        splitted = False
        for name, module in self.model.named_modules():
            if module is self.model.end_conv:
                #print("Can't split feature in end conv")
                continue
            if type(module) is Conv1dExtendable:
                ncc = module.normalized_cross_correlation()
                for feature_number, value in enumerate(ncc):
                    if abs(value.data[0]) > threshold:
                        print("in ", name, ", split feature number ", feature_number)
                        module.split_feature(feature_number=feature_number)
                        splitted = True
        if splitted:
            self.optimizer = self.optimizer_type(params=self.model.parameters(), lr=self.learning_rate)

    def reset_training(self):
        self.i = 0
        self.losses = []
        self.step_times = []
        self.loss_positions = []
        self.validation_results = []
        self.validation_result_positions = []
        self.avg_loss = 0
        self.avg_time = 0
        self.current_epoch = -1

        self.new_epoch()
        self.data.load_new_chunk()
        self.data.use_new_chunk()


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

        self.learning_rate = learning_rate
        self.optimizer.lr = learning_rate
        self.epochs = epochs

        if segments_per_chunk != self.segments_per_chunk | examples_per_segment != self.examples_per_segment:
            self.segments_per_chunk = segments_per_chunk
            self.examples_per_segment = examples_per_segment

            self.new_epoch()
            self.data.load_new_chunk()
            self.data.use_new_chunk()

        self.model.train()  # set to train mode

        # train loop
        while True:
            tic = time.time()
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

            step_time = time.time() - tic
            self.avg_time += step_time
            self.avg_loss += loss
            self.i += 1

            # train feedback
            if self.i % self.report_interval == 0:
                print("loss: ", loss)

                #if self.report_callback != None:
                #    self.report_callback(self)

            if self.i % self.logging_interval == 0:
                self.avg_loss /= self.logging_interval
                self.avg_time /= self.logging_interval
                previous_loss = self.avg_loss

                self.losses.append(self.avg_loss)
                self.step_times.append(self.avg_time)
                self.loss_positions.append(self.i)

                print("log to tensorBoard")
                self.log_to_tensor_board()
                self.split_important_features(threshold=0.2)
                #self.log_normalized_cross_correlation()

                self.avg_loss = 0
                self.avg_time = 0

            # run on validation set
            if self.i % self.validation_interval == 0:
                self.validate_model(self.i)
                print("average step time: ", self.step_times[-1])
                # print("validation loss: ", avg_loss)

            # snapshot
            if self.i % self.snapshot_interval == 0:
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
        self.additional_receptive_field = self.receptive_field - self.target_length + 1 # negative offset to accommodate for the receptive field

        # validation data
        self.validation_inputs = np.array(0)
        self.validation_targets = np.array(0)
        self.validation_index_count = 0
        self.validation_positions = []
        self.validation_segment_length = 0.

        # calculate training data duration
        self.data_length = 0
        self.start_positions = [0]
        for path in paths:
            d = librosa.get_duration(filename=path) * self.sampling_rate
            self.data_length += d
            self.start_positions.append(self.data_length)
        print("total duration of training data: ", self.data_length, " samples")

            # self.start_new_epoch()

    def quantize_data(self, data):
        # mu-law enconding
        mu_x = mu_law_enconding(data, self.classes)
        # quantization
        bins = np.linspace(-1, 1, self.classes)
        quantized = np.digitize(mu_x, bins) - 1
        inputs = bins[quantized[0:-1]]
        targets = quantized[1::]
        return inputs, targets

    def create_validation_set(self, segments=32, examples_per_segment=8):
        '''
        Create validation set from data that will be excluded from all training data
        '''

        self.validation_index_count = segments * examples_per_segment
        self.validation_inputs = self.dtype(self.validation_index_count, 1, self.receptive_field).zero_()
        self.validation_targets = self.ltype(self.validation_index_count, self.target_length).zero_()

        self.validation_segment_length = self.target_length * examples_per_segment
        print("The validation set has a total duration of ", segments * self.validation_segment_length, " s")

        available_segments = int(self.data_length // self.validation_segment_length) - 1 # number of segments that can be chosen from
        validation_offset = int(uniform(0, self.validation_segment_length)) # some random offset
        positions = np.random.choice(available_segments, size=segments, replace=False)
        self.validation_positions = positions * self.validation_segment_length + validation_offset

        duration = self.validation_segment_length + self.additional_receptive_field

        for s in range(segments):
            position = self.validation_positions[s] - self.additional_receptive_field
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
                self.validation_inputs[example_index, :, :] = i[position - self.receptive_field:position]
                self.validation_targets[example_index, :] = t[position - self.target_length:position]

    def start_new_epoch(self, segments_per_chunk, examples_per_segment):
        # wait for loading to finish
        # if self.load_thread != None:
        #     if self.load_thread.is_alive():
        #         self.load_thread.join()

        #print("\n start new epoch")

        self.segments_per_chunk = segments_per_chunk
        self.examples_per_segment = examples_per_segment
        self.training_index_count = segments_per_chunk * examples_per_segment
        self.training_segment_duration = self.target_length * examples_per_segment

        training_offset = uniform(0, self.training_segment_duration)
        available_segments = int(self.data_length // self.training_segment_duration) - 1

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

            # check if this segment overlaps with any validation segment,
            # if yes, then block it
            segment_is_blocked = False
            for validation_position in self.validation_positions:
                train_seg_end = segment_position + self.training_segment_duration
                validation_seg_end = validation_position + self.validation_segment_length
                if (train_seg_end > validation_position) & (segment_position < validation_seg_end):
                    #print("block segment at position ", validation_position)
                    segment_is_blocked = True
                    break

            current_chunk_position += 1

            if segment_is_blocked:
                continue

            duration = self.training_segment_duration + self.additional_receptive_field
            new_data = self.load_segment(segment_position, duration)
            i, t = self.quantize_data(new_data)
            self.loaded_data.append((i, t))

        #self.training_index_count = len(self.loaded_data) * self.examples_per_segment
        self.chunk_position = current_chunk_position
        #print("there are ", len(self.loaded_data), " segments in the newly loaded chunk")
        toc = time.time()
        if toc-tic > 60:
            print("loading this chunk took ", toc-tic, " seconds")

        if len(self.loaded_data) == 0:
            print("Loaded data has length 0?!")


    def use_new_chunk(self):
        #print("use loaded chunk with ", len(self.loaded_data), "segments")

        # wait for loading to finish
        if self.load_thread != None:
            if self.load_thread.is_alive():
                print("Loading the data is slowing the training process down. Maybe you should use less segments per chunk or uncompressed audio files.")
                self.load_thread.join()

        if len(self.loaded_data) == 0:
            print("no data loaded?!")

        if self.training_index_count > self.examples_per_segment * self.segments_per_chunk:
            print("To many training indices ?!")

        self.sample_indices = np.random.permutation(self.training_index_count)
        # TODO sometimes the training index count is way to high, why???
        #print("last training index count: ", self.training_index_count)
        if(len(self.sample_indices) > self.segments_per_chunk * self.examples_per_segment):
            print("training index count too high")
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
                print("index ", position, " is not available in a tensor of size ", self.inputs[segment].size(0))

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
        duration_in_s = duration / self.sampling_rate
        file_index = 0
        while self.start_positions[file_index+1] <= segment_position:
            file_index += 1
            if file_index+1 >= len(self.start_positions):
                print("position ", segment_position, "is not available, fill with ", duration, " zeros \n (this should not have happened!!!)")
                zeros = np.zeros((duration))
                return zeros
        file_path = self.paths[file_index]

        # load from file
        offset = (segment_position - self.start_positions[file_index]) / self.sampling_rate
        new_data, sr = librosa.load(path=file_path,
                                    sr=self.sampling_rate,
                                    mono=True,
                                    offset=offset,
                                    duration=duration_in_s)

        # if the file was not long enough, recursively call this function on the next file to get the remaining duration
        new_loaded_duration = len(new_data)
        if new_loaded_duration < duration:
            new_position = self.start_positions[file_index+1]
            new_duration = duration - new_loaded_duration
            additional_data = self.load_segment(new_position, new_duration)
            new_data = np.append(new_data, additional_data)

        if len(new_data) < duration:
            print("loaded segment is to short: \nexpected ", duration, "samples, but got", len(new_data))

        return new_data