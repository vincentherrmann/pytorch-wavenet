import time
from wavenet_training import WaveNetModel, Optimizer, WaveNetData
import torch
import numpy as np

model = WaveNetModel(num_layers=10,
                     num_blocks=2,
                     num_classes=128,
                     hidden_channels=64)

print("model: ", model)
print("scope: ", model.scope)
print("parameter count", model.parameter_count())

data = WaveNetData('../train_samples/violin.wav',
                   input_length=model.scope,
                   target_length=model.last_block_scope,
                   num_classes=model.num_classes)
start_tensor = data.get_minibatch([0])[0].squeeze()

optimizer = Optimizer(model, mini_batch_size=1, avg_length=20)

print('start training...')
tic = time.time()
optimizer.train(data, epochs=100)
toc = time.time()
print('Training took {} seconds.'.format(toc - tic))

torch.save(model.state_dict(), "../model_parameters/violin_10-2-128-164")

print('generate...')
tic = time.time()
generated = model.generate(start_data=start_tensor, num_generate=100)
toc = time.time()
print('Generating took {} seconds.'.format(toc - tic))

print('generate...')
tic = time.time()
generated = model.fast_generate(100)
toc = time.time()
print('Generating took {} seconds.'.format(toc - tic))
