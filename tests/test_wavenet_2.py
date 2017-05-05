import time
from WaveNetModel2 import WaveNetModel2
from model import Optimizer, WaveNetData
from torch.autograd import Variable
import torch
import numpy as np

model = WaveNetModel2(layers=10,
					  blocks=2,
					  dilation_channels=32,
					  residual_channels=32,
					  skip_channels=32,
					  classes=128)

# out = model.forward(Variable(torch.zeros((1, 1, 2048))))
# print(out)

print("model: ", model)
print("scope: ", model.scope)
#print("parameter count", model.parameter_count())

data = WaveNetData('../train_samples/violin.wav',
				   input_length=model.scope,
				   target_length=model.last_block_scope,
				   num_classes=model.classes)
start_tensor = data.get_minibatch([0])[0].squeeze()

optimizer = Optimizer(model, learning_rate=0.001, mini_batch_size=4, avg_length=2)

generated = model.generate_fast(100, sampled_generation=True)
print(generated)

print('start training...')
tic = time.time()
optimizer.train(data, epochs=100)
toc = time.time()
print('Training took {} seconds.'.format(toc - tic))

generated = model.generate_fast(500)
print(generated)