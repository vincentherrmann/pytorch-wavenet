from time import time
import torch
import numpy as np
from torch.autograd import Variable
from wavenet_training import Model, Optimizer, WavenetData, ConvDilated
from scipy.io import wavfile
import visdom

model = Model(num_time_samples=10000,
              num_blocks=4,
              num_layers=10,
              num_hidden=32,
              num_classes=64)
# torch.save(model, 'untrained_model')
print('model: ', model)
print('scope: ', model.scope)

optimizer = Optimizer(model,
                      stop_threshold=0.1)

data = WavenetData('sine.wav',
                   input_length=model.scope,
                   target_length=model.last_block_scope,
                   num_classes=model.num_classes)
start_tensor = data.get_minibatch([12345])[0].squeeze()


def hook(losses):
    print("loss: ", losses[-1])


optimizer.hook = hook

print('start training...')
# tic = time.time()
optimizer.train(data)
# toc = time.time()
print('Training took {} seconds.'.format(toc - tic))
