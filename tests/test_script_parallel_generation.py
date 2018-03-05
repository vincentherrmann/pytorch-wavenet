from wavenet_model import *
from audio_data import *
from wavenet_training import *
from model_logging import *
import scipy.io.wavfile
import numpy as np
import torch

dtype = torch.FloatTensor
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor

# model = load_to_cpu("../snapshots/turca_model_student_noise_seed_5000")
model = load_to_cpu("../snapshots/turca_model_student_one_stack")
print("model has " + str(model.parameter_count()) + " parameters")

model.output_length = 4000
model.dtype = torch.FloatTensor

generated = model.generate(num_samples=32000)
generated_np = generated.data.squeeze().numpy()
scipy.io.wavfile.write('../latest_generated_clip.wav', rate=16000, data=generated_np.astype(np.float32))