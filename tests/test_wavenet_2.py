import time
from wavenet_model import WaveNetModel
from wavenet_training import *
from torch.autograd import Variable
import torch
import numpy as np

model = WaveNetModel(layers=7,
                     blocks=2,
                     dilation_channels=32,
                     residual_channels=32,
                     skip_channels=32,
                     classes=128)

# out = model.forward(Variable(torch.zeros((1, 1, 2048))))
# print(out)

print("model: ", model)
print("scope: ", model.receptive_field)
# print("parameter count", model.parameter_count())

# data = WaveNetData('../train_samples/violin.wav',
# 				   input_length=model.receptive_field,
# 				   target_length=model.output_length,
# 				   num_classes=model.classes)

# data = ["../train_samples/hihat.wav",
#         "../train_samples/piano.wav",
#         "../train_samples/saber.wav",
#         "../train_samples/violin.wav",
#         "../train_samples/sine.wav",
#         "../train_samples/bach_full.wav",
#         "../train_samples/sapiens.wav"]
        #"/Users/vincentherrmann/Music/Mischa Maisky plays Bach Cello Suite No.1 in G (full).wav"]

data = ["../train_samples/piano.wav"]


data_loader = AudioFileLoader(data,
                              classes=model.classes,
                              receptive_field=model.receptive_field,
                              target_length=model.output_length,
                              dtype=model.dtype,
                              ltype=torch.LongTensor,
                              sampling_rate=44100)


# start_tensor = data.get_minibatch([0])[0].squeeze()
# start_tensor = data_loader.get_wavenet_minibatch(indices=[model.receptive_field],
# 												 receptive_field=model.receptive_field,
# 												 target_length=model.output_length)
optimizer = WaveNetOptimizer(model,
                             data=data_loader,
                             test_segments=4,
                             examples_per_test_segment=2,
                             report_interval=4,
                             test_interval=32)

# optimizer = Optimizer(model, learning_rate=0.001, mini_batch_size=4, avg_length=2)

# generated = model.generate_fast(100,
#                                 first_samples=torch.zeros((1)),
#                                 sampled_generation=True)
# print(generated)

print('start training...')
tic = time.time()
# optimizer.train(data, epochs=100)
optimizer.train(learning_rate=0.001,
                minibatch_size=4,
                epochs=10,
                segments_per_chunk=4,
                examples_per_segment=8)
toc = time.time()
print('Training took {} seconds.'.format(toc - tic))

generated = model.generate_fast(500)
print(generated)
