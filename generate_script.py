from time import time
import torch
import numpy as np

from model import Model, Optimizer, WavenetData

model = Model(num_blocks=2, num_layers=8, num_hidden=32, num_classes=64)
gen = model.generate(100, torch.FloatTensor([0]))
print(gen)