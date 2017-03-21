import numpy as np
import torch

from scipy.io import wavfile

def normalize(data):
    temp = np.float32(data) - np.min(data)
    out = (temp / np.max(temp) - 0.5) * 2
    return out


def make_batch(path):
    data = wavfile.read(path)[1][:, 0]
    #data_ = normalize(data)

    bins = np.linspace(-1, 1, 256)
    # Quantize inputs.
    inputs = np.digitize(data[0:-1], bins, right=False) - 1
    inputs = bins[inputs][None, None, :]

    # Encode targets as ints.
    targets = (np.digitize(data[1::], bins, right=False) - 1)[None, :]
    return inputs, targets

def one_hot(labels, num_labels):
	batch = labels.size(0)
	x = torch.zeros(batch, num_labels).scatter_(1, labels.view(-1, 1), 1.)
	return x

