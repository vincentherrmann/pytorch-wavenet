import time
from WaveNetModel2 import WaveNetModel2
from model import Optimizer, WaveNetData, AudioFileLoader, WaveNetOptimizer
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
print("scope: ", model.receptive_field)
#print("parameter count", model.parameter_count())

data = WaveNetData('../train_samples/violin.wav',
				   input_length=model.receptive_field,
				   target_length=model.output_length,
				   num_classes=model.classes)

data_loader = AudioFileLoader(['../train_samples/bach_full.wav'],
							  classes=128,
							  max_load_duration=2.0,
							  dtype=torch.FloatTensor,
							  ltype=torch.LongTensor,
							  sampling_rate=11025)

#start_tensor = data.get_minibatch([0])[0].squeeze()
# start_tensor = data_loader.get_wavenet_minibatch(indices=[model.receptive_field],
# 												 receptive_field=model.receptive_field,
# 												 target_length=model.output_length)
optimizer = WaveNetOptimizer(model,
							 mini_batch_size=2,
							 report_length=4)

#optimizer = Optimizer(model, learning_rate=0.001, mini_batch_size=4, avg_length=2)

# generated = model.generate_fast(100,
# 								first_samples=torch.zeros((1)),
# 								sampled_generation=True)
#print(generated)

print('start training...')
tic = time.time()
#optimizer.train(data, epochs=100)
optimizer.train(data_loader,
				epochs=10)
toc = time.time()
print('Training took {} seconds.'.format(toc - tic))

generated = model.generate_fast(500)
print(generated)