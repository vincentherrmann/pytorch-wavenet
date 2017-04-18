from time import time
import torch
import numpy as np
from torch.autograd import Variable
from utils import make_batch, one_hot
from model import Model, Optimizer, WavenetData, ConvDilated
from scipy.io import wavfile
import visdom

#vis = visdom.Visdom()
#vis.text('Hello, world!')
#vis.image(np.ones((10, 10, 3)))

#test Wavenet Data
#data = Wavenet_data('voice.wav', input_length=32, target_length=20)
#i1, t1 = data.get_minibatch([11400])
#i2, t2 = data.get_minibatch([6])

####test
module = ConvDilated(num_channels_in=1,
							 num_channels_out=1,
							 kernel_size=2,
							 dilation=2)

module.conv.weight = torch.nn.Parameter(torch.FloatTensor([[[0, 1]]]))
module.train()
w = module.conv.weight

input = torch.linspace(0, 12, steps=13).view(1, 1, 13)
dilated = module(Variable(input))
######

inputs, targets = make_batch('sine.wav')
num_time_samples = inputs.shape[2]

print('create model...')
model = Model(num_time_samples=num_time_samples, num_blocks=1, num_layers=8, num_hidden=32, num_classes=64)
torch.save(model, 'untrained_model')

print('model: ', model)
print('scope: ', model.scope)
print('last_block_scope: ', model.last_block_scope)

#print('generate...')
#generated = model.fast_generate(10000, first_sample=0.0)
#print(generated)


data = WavenetData('sine.wav',
				   input_length=model.scope,
				   target_length=model.last_block_scope,
				   num_classes=model.num_classes)
start_tensor = data.get_minibatch([12345])[0].squeeze()

# print('generate...')
# tic = time()
# generated = model.fast_generate(20000, first_sample=0.1)
# toc = time()
# print('Generating took {} seconds.'.format(toc-tic))
#
# print(generated)
# wavfile.write('untrained_generated.wav', rate=44100, data=np.array(generated))



#test_data = torch.nn.Parameter(torch.FloatTensor(np.linspace(0, model.scope-1, model.scope)).view(1, 1, -1))
#test_res = model.forward(test_data)

optimizer = Optimizer(model)

print('start training...')
tic = time()
optimizer.train(data)
toc = time()
print('Training took {} seconds.'.format(toc-tic))

print('generate...')
tic = time()
#generated = model.generate(start_data=start_tensor, num_generate=20000)
generated = model.fast_generate(40000, first_sample=0.1)
toc = time()
print('Generating took {} seconds.'.format(toc-tic))

print(generated)
wavfile.write('trained_generated.wav', rate=44100, data=np.array(generated))