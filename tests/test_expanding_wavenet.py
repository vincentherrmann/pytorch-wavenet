import time
from wavenet_model import WaveNetModel, ExpandingWaveNetModel
from wavenet_training import *
from torch.autograd import Variable
import torch
import numpy as np

model = ExpandingWaveNetModel(layers=5,
                              blocks=2,
                              dilation_channels=4,
                              residual_channels=4,
                              skip_channels=4,
                              classes=256,
                              output_length=32)

print("model: ", model)
print("scope: ", model.receptive_field)
# print("parameter count", model.parameter_count())

parameters = dict(model.named_parameters())
param_map = {id(v): k for k, v in parameters.items()}

test_input = Variable(torch.rand(1, 1, model.receptive_field + model.output_length)) * 2 - 1

test_output_1 = model(test_input)
print("test output 1: ", test_output_1)

model.dilated_convs[2].split_feature(feature_number=1)
#model.skip_convs[3].split_feature(feature_number=3)
# model.skip_convs[2].split_feature(feature_number=1)
# model.skip_convs[6].split_feature(feature_number=5)
#model.residual_convs[3].split_feature(feature_number=3)


test_output_2 = model(test_input)
print("test output 2: ", test_output_2)

diff = test_output_1 - test_output_2
dot = torch.dot(diff, diff)
print("mse: ", dot.data[0])

data = ["../train_samples/piano.wav"]

data_loader = AudioFileLoader(data,
                              classes=model.classes,
                              receptive_field=model.receptive_field,
                              target_length=model.output_length,
                              dtype=model.dtype,
                              ltype=torch.LongTensor,
                              sampling_rate=44100)

optimizer = WaveNetOptimizer(model,
                             data=data_loader,
                             optimizer=optim.Adam,
                             validation_segments=4,
                             examples_per_validation_segment=2,
                             report_interval=4,
                             validation_interval=256,
                             logging_interval=256,
                             segments_per_chunk=4,
                             examples_per_segment=8)

optimizer.log_normalized_cross_correlation()

print('start training...')
tic = time.time()
optimizer.train(learning_rate=0.001,
                minibatch_size=4,
                epochs=100,
                segments_per_chunk=4,
                examples_per_segment=8)
toc = time.time()
print('Training took {} seconds.'.format(toc - tic))

generated = model.generate_fast(500)
print(generated)
