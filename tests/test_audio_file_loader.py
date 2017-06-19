import time
import torch
import time
import numpy as np
import librosa
from wavenet_model import WaveNetModel
from wavenet_training import *
from unittest import TestCase



class Test_AudioFileLoader(TestCase):
    def __init__(self):
        data = ["../train_samples/hihat.wav",
                "../train_samples/piano.wav",
                "../train_samples/saber.wav",
                "../train_samples/violin.wav",
                "../train_samples/sine.wav",
                "../train_samples/bach_full.wav",
                "../train_samples/sapiens.wav",
                "/Users/vincentherrmann/Music/Mischa Maisky plays Bach Cello Suite No.1 in G (full).wav"]

        self.audio_loader = AudioFileLoader(data,
                                            classes=256,
                                            receptive_field=2000,
                                            target_length=1000,
                                            dtype=torch.FloatTensor,
                                            ltype=torch.LongTensor,
                                            sampling_rate=11025,
                                            epoch_finished_callback=self.start_new_epoch)


    def test_set(self):
        self.audio_loader.create_validation_set(segments=8, examples_per_segment=8)
        print("test inputs: ", self.audio_loader.validation_inputs)
        print("test targets: ", self.audio_loader.validation_targets)

        self.audio_loader.start_new_epoch(segments_per_chunk=16,
                                          examples_per_segment=32)
        self.audio_loader.load_new_chunk()
        self.audio_loader.use_new_chunk()

        # minibatch = self.audio_loader.get_minibatch(32)
        # print("first minibatch: ", minibatch)
        #
        # print("")
        # print("############")
        # print("")

        for i in range(100):
            minibatch = self.audio_loader.get_minibatch(16)
            print("    minibatch", i, " loaded")

        # test_sample, t = self.audio_loader.quantize_data(self.audio_loader.test_inputs.numpy()[0, 0, 1:2])
        # assert test_sample == self.audio_loader.test_targets[0, 0]
        #
        # test_sample = mu_law_econding(self.audio_loader.test_inputs[-1, 0, -1], mu=self.audio_loader.classes)
        # assert test_sample== self.audio_loader.test_targets[-1, -2]

    def start_new_epoch(self):
        print("start new epoch")
        self.audio_loader.start_new_epoch(segments_per_chunk=16,
                                          examples_per_segment=32)

test = Test_AudioFileLoader()
test.test_set()