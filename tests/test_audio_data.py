from unittest import TestCase
import cProfile
import time
from audio_data import *
import numpy as np
import torch.utils.data

class TestWavenetDataset(TestCase):
    def test_dataset_creation(self):
        in_path = '../train_samples'
        out_path = '../train_samples/test_dataset.npz'
        dataset = WavenetDataset(dataset_file=out_path,
                                 item_length=1000,
                                 target_length=64,
                                 file_location=in_path)
        item0 = dataset[len(dataset)-3]
        item1 = dataset[len(dataset)-2]
        item2 = dataset[len(dataset)-1]

        assert item0[0][0, -33] == 2. * item0[1][0, -34] / dataset.classes - 1.
        assert item1[0][0, -33] == 2. * item1[1][0, -34] / dataset.classes - 1.
        assert item2[0][0, -33] == 2. * item2[1][0, -34] / dataset.classes - 1.

        assert 2. * item0[1][0, -1] / dataset.classes - 1. == item1[0][0, -dataset.target_length]
        assert 2. * item1[1][0, -1] / dataset.classes - 1. == item2[0][0, -dataset.target_length]

    def test_minibatch_performance(self):
        dataset = WavenetDataset(dataset_file='../train_samples/test_dataset.npz',
                                 item_length=1000,
                                 target_length=64)
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=32,
                                                 shuffle=True,
                                                 num_workers=8)
        dataloader_iter = iter(dataloader)
        num_batches = 10

        def calc_batches(num=1):
            for i in range(num):
                mb = next(dataloader_iter)
            return mb

        tic = time.time()
        last_minibatch = calc_batches(num_batches)
        toc = time.time()

        print("time it takes to calculate "  + str(num_batches) + " minibatches: " + str(toc-tic) + " s")
        assert False




class TestListAllAudioFiles(TestCase):
    def test_list_all_audio_files(self):
        files = list_all_audio_files('../train_samples')
        print(files)
        assert len(files) > 0


class TestQuantizeData(TestCase):
    def test_quantize_data(self):
        data = np.random.rand(32) * 2 - 1
        qd = quantize_data(data, 256)
        print(qd)
        print(qd.dtype)
        assert False



