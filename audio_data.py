import os
import os.path
import math
import threading
import torch
import torch.utils.data
import numpy as np
import librosa as lr
import bisect


class WavenetDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_file,
                 item_length,
                 file_location=None,
                 classes=256,
                 sampling_rate=16000,
                 mono=True,
                 normalize=False,
                 dtype=np.uint8,
                 input_type=torch.FloatTensor,
                 target_type=torch.IntTensor):

        self.dataset_file = dataset_file
        self._item_length = item_length

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
            self.classes = classes
            self.sampling_rate = None
            self.dtype = None

        self.input_type = input_type
        self.target_type = target_type
        self.data = np.load(self.dataset_file, mmap_mode='r')
        self.start_samples = [0]
        self._length = 0
        self.calculate_length()

    def create_dataset(self, location, out_file):
        print("create dataset from audio files at", location)
        self.dataset_file = out_file
        files = list_all_audio_files(location)
        processed_files = []
        for i, file in enumerate(files):
            print("  processed " + str(i) + " of " + str(len(files)) + " files")
            file_data, _ = lr.load(path=file,
                                   sr=self.sampling_rate,
                                   mono=self.mono)
            if self.normalize:
                file_data = lr.util.normalize(file_data)
            quantized_data = quantize_data(file_data, self.classes).astype(self.dtype)
            processed_files.append(quantized_data)

        np.savez(self.dataset_file, *processed_files)

    def calculate_length(self):
        start_samples = [0]
        for i in range(len(self.data.keys())):
            start_samples.append(start_samples[-1] + len(self.data['arr_' + str(i)]))
        self._length = math.floor((start_samples[-1]-1) / self._item_length)
        self.start_samples = start_samples
        self.start_samples[0] = -1

    def set_item_length(self, l):
        self._item_length = l
        self.calculate_length()

    def __getitem__(self, idx):
        sample_index = idx * self._item_length
        file_index = bisect.bisect_left(self.start_samples, sample_index) - 1
        position_in_file = sample_index - self.start_samples[file_index]
        end_position_in_next_file = sample_index + self._item_length + 1 - self.start_samples[file_index + 1]

        if end_position_in_next_file < 0:
            file_name = 'arr_' + str(file_index)
            this_file = np.load(self.dataset_file, mmap_mode='r')[file_name]
            sample = this_file[position_in_file:position_in_file + self._item_length + 1]
            #return sample[:self._item_length], sample[1:]
        else:
            # load from two files
            file1 = np.load(self.dataset_file, mmap_mode='r')['arr_' + str(file_index)]
            file2 = np.load(self.dataset_file, mmap_mode='r')['arr_' + str(file_index + 1)]
            sample1 = file1[position_in_file:]
            sample2 = file2[:end_position_in_next_file]
            sample = np.concatenate((sample1, sample2))
            #return sample[:self._item_length], sample[1:]

        example = torch.from_numpy(sample[:self._item_length]).type(self.input_type).unsqueeze(0)
        example = 2. * example / self.classes - 1.
        target = torch.from_numpy(sample[1:]).type(self.target_type).unsqueeze(0)
        return example, target

    def __len__(self):
        return self._length


def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    bins = np.linspace(-1, 1, classes)
    quantized = np.digitize(mu_x, bins) - 1
    return quantized


def list_all_audio_files(location):
    audio_files = []
    for dirpath, dirnames, filenames in os.walk(location):
        for filename in [f for f in filenames if f.endswith((".mp3", ".wav", ".aif", "aiff"))]:
            audio_files.append(os.path.join(dirpath, filename))

    return audio_files


def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x


def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s