from time import time
import torch
import numpy as np
import time

from model import Model, Optimizer, WavenetData

model = Model(num_time_samples=10000,
              num_blocks=1,
              num_layers=10,
              num_hidden=32,
              num_classes=64)

model.load_state_dict(torch.load("trained_state"))

data = WavenetData('sine.wav',
                   input_length=model.scope,
                   target_length=model.last_block_scope,
                   num_classes=model.num_classes)

#start_tensor = data.get_minibatch([12345])[0].squeeze()
start_tensor = torch.zeros((model.scope)) + 0.0

# print('generate...')
# tic = time.time()
# generated = model.generate(start_data=start_tensor, num_generate=200)
# toc = time.time()
# print('Generating took {} seconds.'.format(toc-tic))

# print('generate...')
# tic = time.time()
# [generated, support_generated] = model.fast_generate(200, first_samples=torch.zeros((1)))
# toc = time.time()
# print('Generating took {} seconds.'.format(toc-tic))

print('generate...')
tic = time.time()
[conv_generated, fast_generated] = model.compare_generate(200, first_samples=start_tensor)
toc = time.time()
print('Generating took {} seconds.'.format(toc-tic))