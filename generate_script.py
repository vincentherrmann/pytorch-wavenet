import time
import librosa.output
from wavenet_model import *

from IPython.display import Audio
from IPython.core.debugger import Tracer
from matplotlib import pyplot as plt
from matplotlib import pylab as pl
from IPython import display
import torch
import numpy as np

model = WaveNetModel(layers=6,
                     blocks=4,
                     dilation_channels=16,
                     residual_channels=16,
                     skip_channels=16,
                     output_length=8,
                     dtype=torch.FloatTensor)

model = load_latest_model_from('snapshots', use_cuda=False)
model.cpu()
model.dtype = torch.FloatTensor
for q in model.dilated_queues:
    q.dtype = torch.FloatTensor

#model = torch.load('snapshots/snapshot_2017-12-10_10-30-14')

data = WavenetDataset(dataset_file='train_samples/saber/dataset.npz',
                      item_length=model.receptive_field + model.output_length - 1,
                      target_length=model.output_length,
                      file_location='train_samples/saber',
                      test_stride=20)
print('the dataset has ' + str(len(data)) + ' items')

start_data = data[10000][0]
start_data = torch.max(start_data, 0)[1]

print("generate")
tic = time.time()
generated = model.generate_fast(num_samples=4000,
                                first_samples=start_data,
                                temperature=1.)
toc = time.time()
print("generating took " + str(toc-tic) + " s")

print(generated)
librosa.output.write_wav('latest_generated_clip.wav', generated, sr=16000)
#
# model = Model(num_time_samples=10000,
#               num_blocks=2,
#               num_layers=11,
#               num_hidden=128,
#               num_classes=256)
# print('model: ', model)
# print('scope: ', model.scope)
#
# data = WavenetData('bach_11025.wav',
#                    input_length=model.scope,
#                    target_length=model.last_block_scope,
#                    num_classes=model.num_classes)
# start_tensor = data.get_minibatch([30000])[0].squeeze()
#
# optimizer = Optimizer(model,
#                       learning_rate=0.003,
#                       stop_threshold=0.1,
#                       avg_length=4)
#
# print('start training...')
# tic = time.time()
# optimizer.train(data)
# toc = time.time()
# print('Training took {} seconds.'.format(toc - tic))
#
# model.load_state_dict(torch.load("trained_state_piano"))
#
# data = WavenetData('sine.wav',
#                    input_length=model.scope,
#                    target_length=model.last_block_scope,
#                    num_classes=model.num_classes)
#
# # start_tensor = data.get_minibatch([12345])[0].squeeze()
# start_tensor = torch.zeros((model.scope)) + 0.0
#
# # print('generate...')
# # tic = time.time()
# # generated = model.generate(start_data=start_tensor, num_generate=200)
# # toc = time.time()
# # print('Generating took {} seconds.'.format(toc-tic))
#
# print('generate...')
# tic = time.time()
# [generated, support_generated] = model.fast_generate(200, first_samples=torch.zeros((1)))
# toc = time.time()
# print('Generating took {} seconds.'.format(toc - tic))
#
# # print('generate...')
# # tic = time.time()
# # [conv_generated, fast_generated] = model.compare_generate(200, first_samples=start_tensor)
# # toc = time.time()
# # print('Generating took {} seconds.'.format(toc-tic))
