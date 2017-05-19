from wavenet_training import WaveNetData
import torch

data = WaveNetData('../train_samples/sine.wav',
                   input_length=20,
                   target_length=15,
                   num_classes=16)

minibatch = data.get_minibatch([5, 500, 5000])
print(minibatch)
