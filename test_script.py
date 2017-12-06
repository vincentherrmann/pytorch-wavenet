import time
import torch
import numpy as np
from torch.autograd import Variable
from wavenet_model import WaveNetModel
from audio_data import WavenetDataset
from wavenet_training import WavenetOptimizer
from scipy.io import wavfile

model = WaveNetModel(layers=10,
                     blocks=4)

in_path = '../train_samples'
out_path = '../train_samples/test_dataset.npz'
data = WavenetDataset(dataset_file='train_samples/test_dataset.npz',
                      item_length=model.receptive_field,
                      file_location='../train_samples')

# torch.save(model, 'untrained_model')
print('model: ', model)
print('receptive field: ', model.receptive_field)

optimizer = WavenetOptimizer(model=model,
                             dataset=data,
                             lr=0.001)

print('start training...')
tic = time.time()
optimizer.train(batch_size=11,
                epochs=1)
toc = time.time()
print('Training took {} seconds.'.format(toc - tic))
