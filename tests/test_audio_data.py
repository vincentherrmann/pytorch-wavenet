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
                                 item_length=64,
                                 file_location=in_path)
        item0 = dataset[len(dataset)-3]
        item1 = dataset[len(dataset)-2]
        item2 = dataset[len(dataset)-1]

        assert item0[0][34] == item0[1][33]
        assert item1[0][34] == item1[1][33]
        assert item2[0][34] == item2[1][33]

        assert item0[1][-1] == item1[0][0]
        assert item1[1][-1] == item2[0][0]

    def test_minibatch_performance(self):
        dataset = WavenetDataset(dataset_file='../train_samples/test_dataset.npz',
                                           item_length=64)
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


class TestCreateWavenetDataset(TestCase):
    def setUp(self):
        self.in_path = '../train_samples'
        self.out_path = '../train_samples/test_dataset.npz'

    def test_create_wavenet_dataset(self):
        in_path = '../train_samples'
        out_path = '../train_samples/test_dataset.npz'
        create_wavenet_dataset(in_location=in_path, out_location=out_path)
        assert True

    def test_load_dataset(self):
        out_path = '../train_samples/test_dataset.npz'
        file = np.load(out_path)
        first_track = file['arr_0']
        assert first_track.dtype == np.uint8
        assert len(first_track) > 10000
        assert len(first_track) < 10000000



