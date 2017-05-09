import time

from wavenet_training import Model, Optimizer, WaveNetData

from IPython.display import Audio
from IPython.core.debugger import Tracer
from matplotlib import pyplot as plt
from matplotlib import pylab as pl
from IPython import display
import torch
import numpy as np

model = Model(num_time_samples=10000,
              num_blocks=2,
              num_layers=11,
              num_hidden=128,
              num_classes=256)
print('model: ', model)
print('scope: ', model.scope)

data = WavenetData('bach_11025.wav',
                   input_length=model.scope,
                   target_length=model.last_block_scope,
                   num_classes=model.num_classes)
start_tensor = data.get_minibatch([30000])[0].squeeze()

optimizer = Optimizer(model,
                      learning_rate=0.003,
                      stop_threshold=0.1,
                      avg_length=4)

print('start training...')
tic = time.time()
optimizer.train(data)
toc = time.time()
print('Training took {} seconds.'.format(toc - tic))

model.load_state_dict(torch.load("trained_state_piano"))

data = WavenetData('sine.wav',
                   input_length=model.scope,
                   target_length=model.last_block_scope,
                   num_classes=model.num_classes)

# start_tensor = data.get_minibatch([12345])[0].squeeze()
start_tensor = torch.zeros((model.scope)) + 0.0

# print('generate...')
# tic = time.time()
# generated = model.generate(start_data=start_tensor, num_generate=200)
# toc = time.time()
# print('Generating took {} seconds.'.format(toc-tic))

print('generate...')
tic = time.time()
[generated, support_generated] = model.fast_generate(200, first_samples=torch.zeros((1)))
toc = time.time()
print('Generating took {} seconds.'.format(toc - tic))

# print('generate...')
# tic = time.time()
# [conv_generated, fast_generated] = model.compare_generate(200, first_samples=start_tensor)
# toc = time.time()
# print('Generating took {} seconds.'.format(toc-tic))
