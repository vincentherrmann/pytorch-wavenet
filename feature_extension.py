import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.io import wavfile
from random import randint
from wavenet_modules import *

