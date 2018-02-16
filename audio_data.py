import os
import os.path
import math
import threading
import torch
import torch.utils.data
import numpy as np
import librosa as lr
import bisect
import h5py
import scipy
import soundfile
import time
from torch.autograd import Variable
from pathlib import Path


class WavenetMixtureDataset(torch.utils.data.Dataset):
    def __init__(self,
                 location,
                 item_length,
                 target_length,
                 sampling_rate=16000,
                 mono=True,
                 test_stride=100,
                 create_files=True):

        self.location = Path(location)
        self.dataset_path = self.location / 'dataset'
        self.target_length = target_length
        self.sampling_rate = sampling_rate
        self.mono=mono
        self._item_length = item_length
        self._test_stride = test_stride
        self._length = 0
        self.start_samples = [0]
        self.train = True

        if create_files:
            if self.dataset_path.exists():
                self.files = list_all_audio_files(self.dataset_path)
            else:
                unprocessed_files = list_all_audio_files(self.location)
                self.dataset_path.mkdir()
                self.create_dataset(unprocessed_files)
                self.files = list_all_audio_files(self.dataset_path)
        else:
            self.files = list_all_audio_files(self.location)

        self.calculate_length()

    def load_file(self, file, frames=-1, start=0):
        if self.create_files:
            data, _ = soundfile.read(file, frames, start, dtype='float32')
        else:
            data, _ = lr.load(file,
                              sr=self.sampling_rate,
                              mono=self.mono,
                              dtype=np.float32)
            data = data[start:start+frames]
        return data

    def create_dataset(self, files):
        for i, file in enumerate(files):
            data, _ = lr.load(str(file), sr=self.sampling_rate, mono=self.mono, dtype=np.float32)
            new_name = 'file_' + str(i) + ".wav"
            new_file = self.dataset_path / new_name
            lr.output.write_wav(str(new_file), data, sr=self.sampling_rate)
            print("processed " + str(file))

    def calculate_length(self):
        """
        Calculate the number of items in this data sets.
        Additionally the start positions of each file are calculate in this method.
        """
        start_samples = [0]
        for idx in range(len(self.files)):
            file_data, _ = lr.load(str(self.files[idx]),
                                   sr=self.sampling_rate,
                                   mono=self.mono,
                                   dtype=np.float32)
            start_samples.append(start_samples[-1] + file_data.size)
        available_length = start_samples[-1] - (self._item_length - (self.target_length - 1)) - 1
        self._length = math.floor(available_length / self.target_length)
        self.start_samples = start_samples

    def set_item_length(self, l):
        self._item_length = l
        self.calculate_length()

    def load_sample(self, file_index, position_in_file, item_length):
        file_data, _ = lr.load(str(self.files[file_index]),
                       sr=self.sampling_rate,
                       mono=self.mono,
                       dtype=np.float32)
        remaining_length = position_in_file + item_length + 1 - len(file_data)
        if remaining_length < 0:
            sample = file_data[position_in_file:position_in_file + item_length + 1]
        else:
            this_sample = file_data[position_in_file:]
            next_sample = self.load_sample(file_index + 1,
                                           position_in_file=0,
                                           item_length=remaining_length)
            sample = np.concatenate((this_sample, next_sample))
        return sample

    def get_position(self, idx):
        """
        Calculate the file and the position in the file from the global dataset index
        """
        if self._test_stride < 2:
            sample_index = idx * self.target_length
        elif self.train:
            sample_index = idx * self.target_length + math.floor(idx / (self._test_stride-1)) * self.target_length
        else:
            sample_index = self.target_length * (self._test_stride * (idx+1) - 1)

        file_index = bisect.bisect_left(self.start_samples, sample_index) - 1
        if file_index < 0:
            file_index = 0
        if file_index + 1 >= len(self.start_samples):
            print("error: sample index " + str(sample_index) + " is to high. Results in file_index " + str(file_index))
        position_in_file = sample_index - self.start_samples[file_index]
        return file_index, position_in_file

    def __getitem__(self, idx):
        file_index, position_in_file = self.get_position(idx)
        sample = self.load_sample(file_index, position_in_file, self._item_length)

        example = torch.from_numpy(sample[:self._item_length]).type(torch.FloatTensor).unsqueeze(0)
        target = torch.from_numpy(sample[-self.target_length:]).type(torch.FloatTensor).unsqueeze(0)
        return example, target

    def get_segment(self, position=0, file_index=0, duration=None):
        """
        Convenience function to get a segment from a file
        :param position: position in the file in seconds
        :param file_index: index of the file
        :param duration: the duration of the segment in seconds (plus the receptive field). If 'None', then only one receptive field is returned.
        :return: the specified segment (without labels)
        """
        position_in_file = (position // self.sampling_rate) - self.start_samples[file_index]
        if duration is None:
            item_length = self._item_length
        else:
            item_length = int(duration * self.sampling_rate)
        segment = self.load_sample(file_index, position_in_file, item_length)
        return segment

    def __len__(self):
        test_length = math.floor(self._length / self._test_stride)
        if self.train:
            return self._length - test_length
        else:
            return test_length

    @staticmethod
    def process_batch(batch, dtype, ltype):
        example, target = batch
        example = Variable(example.type(dtype))
        target = Variable(target.type(ltype))
        return example, target


class WavenetDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_file,
                 item_length,
                 target_length,
                 file_location=None,
                 classes=256,
                 sampling_rate=16000,
                 mono=True,
                 normalize=False,
                 dtype=np.uint8,
                 train=True,
                 test_stride=100):

        #           |----receptive_field----|
        #                                 |--output_length--|
        # example:  | | | | | | | | | | | | | | | | | | | | |
        # target:                           | | | | | | | | | |

        self.dataset_file = dataset_file
        self._item_length = item_length
        self._test_stride = test_stride
        self.target_length = target_length
        self.classes = classes

        if not os.path.isfile(dataset_file):
            assert file_location is not None, "no location for dataset files specified"
            self.mono = mono
            self.normalize = normalize

            self.sampling_rate = sampling_rate
            self.dtype = dtype
            self.create_dataset(file_location, dataset_file)
        else:
            # Unknown parameters of the stored dataset
            # TODO Can these parameters be stored, too?
            self.mono = None
            self.normalize = None

            self.sampling_rate = None
            self.dtype = None

        self.data = np.load(self.dataset_file, mmap_mode='r')
        self.start_samples = [0]
        self._length = 0
        self.calculate_length()
        self.train = train
        # print("one hot input")
        # assign every *test_stride*th item to the test set

    def create_dataset(self, location, out_file):
        print("create dataset from audio files at", location)
        self.dataset_file = out_file
        files = list_all_audio_files(location)

        # processed_files = []
        # for i, file in enumerate(files):
        #     print("  processed " + str(i) + " of " + str(len(files)) + " files")
        #     file_data, _ = lr.load(path=file,
        #                            sr=self.sampling_rate,
        #                            mono=self.mono)
        #     if self.normalize:
        #         file_data = lr.util.normalize(file_data)
        #     quantized_data = quantize_data(file_data, self.classes).astype(self.dtype)
        #     processed_files.append(quantized_data)
        #
        # np.savez(self.dataset_file, *processed_files)

        processed_files = {}
        for i, file in enumerate(files):
            this_dict = self.process_file(file, i)
            processed_files = {**processed_files, **this_dict}
            print("  processed " + str(i) + " of " + str(len(files)) + " files")
        np.savez(self.dataset_file, **processed_files)

    def process_file(self, path, index):
        file_data, _ = lr.load(path=path,
                               sr=self.sampling_rate,
                               mono=self.mono)
        if self.normalize:
            file_data = lr.util.normalize(file_data)

        quantized_data = quantize_data(file_data, self.classes).astype(self.dtype)
        return {"file_" + str(index): quantized_data}

    def calculate_length(self):
        start_samples = [0]
        for k in self.data.keys():
            if "file_" in k:
                start_samples.append(start_samples[-1] + len(self.data[k]))
        available_length = start_samples[-1] - (self._item_length - (self.target_length - 1)) - 1
        self._length = math.floor(available_length / self.target_length)
        self.start_samples = start_samples

    def set_item_length(self, l):
        self._item_length = l
        self.calculate_length()

    def load_sample(self, file_index, position_in_file, item_length):
        file_name = 'file_' + str(file_index)
        this_file = np.load(self.dataset_file, mmap_mode='r')[file_name]
        remaining_length = position_in_file + item_length + 1 - len(this_file)
        if remaining_length < 0:
            sample = this_file[position_in_file:position_in_file + item_length + 1]
        else:
            this_sample = this_file[position_in_file:]
            next_sample = self.load_sample(file_index + 1,
                                           position_in_file=0,
                                           item_length=remaining_length)
            sample = np.concatenate((this_sample, next_sample))
        return sample

    def get_position(self, idx):
        if self._test_stride < 2:
            sample_index = idx * self.target_length
        elif self.train:
            sample_index = idx * self.target_length + math.floor(idx / (self._test_stride-1)) * self.target_length
            # print("train sample index: ", sample_index)
        else:
            sample_index = self.target_length * (self._test_stride * (idx+1) - 1)
            # print("test sample index: ", sample_index)

        file_index = bisect.bisect_left(self.start_samples, sample_index) - 1
        if file_index < 0:
            file_index = 0
        if file_index + 1 >= len(self.start_samples):
            print("error: sample index " + str(sample_index) + " is to high. Results in file_index " + str(file_index))
        position_in_file = sample_index - self.start_samples[file_index]
        return file_index, position_in_file

    def get_segment(self, position=0, file_index=0, duration=None):
        """
        Get a segment from a file
        :param position: position in the file in seconds
        :param file_index: index of the file
        :param duration: the duration of the segment in seconds (plus the receptive field). If 'None', then only one receptive field is returned.
        :return: the specified segment (without labels)
        """
        position_in_file = (position // self.sampling_rate) - self.start_samples[file_index]
        if duration is None:
            item_length = self._item_length
        else:
            item_length = int(duration * self.sampling_rate)
        segment = self.load_sample(file_index, position_in_file, item_length)
        return segment

    def __getitem__(self, idx):
        file_index, position_in_file = self.get_position(idx)
        sample = self.load_sample(file_index, position_in_file, self._item_length)

        # ONE_HOT:
        # example = torch.from_numpy(sample).type(torch.LongTensor)
        # one_hot = torch.FloatTensor(self.classes, self._item_length).zero_()
        # one_hot.scatter_(0, example[:self._item_length].unsqueeze(0), 1.)
        # target = example[-self.target_length:].unsqueeze(0)
        # return one_hot, target

        example = torch.from_numpy(sample[:self._item_length]).type(torch.FloatTensor).unsqueeze(0)
        example = 2. * example / self.classes - 1.
        target = torch.from_numpy(sample[-self.target_length:]).type(torch.LongTensor).unsqueeze(0)
        return example, target

    def __len__(self):
        test_length = math.floor(self._length / self._test_stride)
        if self.train:
            return self._length - test_length
        else:
            return test_length


class WavenetDatasetWithRandomConditioning(WavenetDataset):
    def __init__(self, *args, **kwargs):
        try:
            conditioning_period = kwargs.pop('conditioning_period')
        except KeyError:
            conditioning_period = 128

        try:
            conditioning_breadth = kwargs.pop('conditioning_breadth')
        except KeyError:
            conditioning_breadth = 8 # in seconds!

        try:
            conditioning_channels = kwargs.pop('conditioning_channels')
        except KeyError:
            conditioning_channels = 16

        self.conditioning_period = conditioning_period
        self.conditioning_breadth = conditioning_breadth
        self.conditioning_channels = conditioning_channels

        super().__init__(*args, **kwargs)

    def process_file(self, path, index):
        super_dict = super().process_file(path, index)

        # zero pad file data to make it compatible with the conditioning period
        file_data = list(super_dict.values())[0]
        file_length = len(file_data)
        pad_length = file_length % self.conditioning_period
        file_data = np.pad(file_data, (0, pad_length), 'constant')
        super_dict[list(super_dict.keys())[0]] = file_data

        # Create smooth random walk as conditioning
        random_length = file_length // (self.sampling_rate * self.conditioning_breadth) + 3
        conditioning_length = file_length // self.conditioning_period
        random_period = self.sampling_rate * self.conditioning_breadth // self.conditioning_period

        random_values = np.random.rand(self.conditioning_channels, random_length)
        conditioning_values = np.zeros([self.conditioning_channels, random_length * random_period])
        x = np.arange(0, random_length)
        xs = np.linspace(0, random_length, conditioning_values.shape[1])
        for c in range(self.conditioning_channels):
            spl = scipy.interpolate.UnivariateSpline(x, random_values[c])
            spl.set_smoothing_factor(1.)
            conditioning_values[c] = spl(xs)
        # cut away beginning and end to avoid irregularities from the smoothing spline
        conditioning_values = conditioning_values[:, random_period:random_period+conditioning_length]

        if np.amax(np.abs(conditioning_values), axis=(0,1)) > 1.2:
            print("irregularity in conditioning values, max value greater than 1.2")

        conditioning_values.astype('float')
        conditioning_dict = {"conditioning_" + str(index): conditioning_values}
        return {**super_dict, **conditioning_dict}

    def load_sample(self, file_index, position_in_file, item_length):
        file_name = 'file_' + str(file_index)
        cond_name = 'conditioning_' + str(file_index)
        audio_file = np.load(self.dataset_file, mmap_mode='r')[file_name]
        conditioning_file = np.load(self.dataset_file, mmap_mode='r')[cond_name]

        conditioning_offset = position_in_file % self.conditioning_period
        c_position_in_file = position_in_file // self.conditioning_period
        c_item_length = item_length // self.conditioning_period + 1

        remaining_length = position_in_file + item_length + 1 - len(audio_file)

        if remaining_length < 0:
            sample = audio_file[position_in_file:position_in_file + item_length + 1]
            conditioning = conditioning_file[:, c_position_in_file:c_position_in_file + c_item_length + 1]
        else:
            this_sample = audio_file[position_in_file:]
            this_conditioning = conditioning_file[c_position_in_file:]
            next_sample, next_conditioning, _ = self.load_sample(file_index + 1,
                                                                 position_in_file=0,
                                                                 item_length=remaining_length)
            sample = np.concatenate((this_sample, next_sample))
            conditioning = np.concatenate((this_conditioning, next_conditioning), axis=1)
        if conditioning.shape[1] != c_item_length + 1:
            print("conditioning has the wrong length!")
        return sample, conditioning, conditioning_offset

    def get_segment(self, position=0, file_index=0, duration=None):
        """
        Get a segment from a file
        :param position: position in the file in seconds
        :param file_index: index of the file
        :param duration: the duration of the segment in seconds (plus the receptive field). If 'None', then only one receptive field is returned.
        :return: the specified segment (without labels)
        """
        position_in_file = (position // self.sampling_rate) - self.start_samples[file_index]
        if duration is None:
            item_length = self._item_length
        else:
            item_length = int(duration * self.sampling_rate)
        sample, conditioning, offset = self.load_sample(file_index, position_in_file, item_length)
        return sample, conditioning, offset

    def __getitem__(self, idx):
        file_index, position_in_file = self.get_position(idx)
        sample, conditioning, offset = self.load_sample(file_index, position_in_file, self._item_length)

        # ONE_HOT:
        # example = torch.from_numpy(sample).type(torch.LongTensor)
        # one_hot = torch.FloatTensor(self.classes, self._item_length).zero_()
        # one_hot.scatter_(0, example[:self._item_length].unsqueeze(0), 1.)
        # target = example[-self.target_length:].unsqueeze(0)
        # return one_hot, target

        example = torch.from_numpy(sample[:self._item_length]).type(torch.FloatTensor).unsqueeze(0)
        example = 2. * example / self.classes - 1.
        conditioning = torch.from_numpy(conditioning).type(torch.FloatTensor)
        target = torch.from_numpy(sample[-self.target_length:]).type(torch.LongTensor).unsqueeze(0)
        return example, conditioning, offset, target

    @staticmethod
    def process_batch(batch, dtype, ltype):
        example, conditioning, offset, target = batch
        example = Variable(example.type(dtype))
        conditioning = Variable(conditioning.type(dtype), volatile=False)
        target = Variable(target.type(ltype))
        return (example, conditioning, offset), target


class WavenetDatasetWithSineConditioning(WavenetDatasetWithRandomConditioning):
    # conditioning_breadth is here the length of the sequence in seconds that will have unique conditioning
    def process_file(self, path, index):
        super_dict = WavenetDataset.process_file(self, path, index)

        # zero pad file data to make it compatible with the conditioning period
        file_data = list(super_dict.values())[0]
        file_length = len(file_data)
        pad_length = file_length % self.conditioning_period
        file_data = np.pad(file_data, (0, pad_length), 'constant')
        super_dict[list(super_dict.keys())[0]] = file_data

        # Create sine curves with
        conditioning_period = file_length / (self.sampling_rate * self.conditioning_breadth)
        conditioning_length = file_length // self.conditioning_period + 3

        conditioning_values = np.zeros([self.conditioning_channels, conditioning_length])
        x = np.linspace(0, np.pi * conditioning_period, conditioning_length)
        for c in range(self.conditioning_channels):
            conditioning_values[c, :] = np.cos(x * (c + 1))

        conditioning_values.astype('float')
        conditioning_dict = {"conditioning_" + str(index): conditioning_values}
        return {**super_dict, **conditioning_dict}


def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    bins = np.linspace(-1, 1, classes)
    quantized = np.digitize(mu_x, bins) - 1
    return quantized


def omit_conditioning(batch, dtype, ltype):
  example, conditioning, offset, target = batch
  example = Variable(example.type(dtype))
  target = Variable(target.type(ltype))
  return example, target


def list_all_audio_files(location):
    types = [".mp3", ".wav", ".aif", "aiff"]
    audio_files = []
    for type in types:
        audio_files.extend(sorted(location.glob('**/*' + type)))
    if len(audio_files) == 0:
        print("found no audio files in " + str(location))
    return audio_files


def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x


def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s