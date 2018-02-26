from wavenet_model import *
from audio_data import *
from wavenet_training import *
from model_logging import *
import scipy.io.wavfile
import numpy as np

dtype = torch.FloatTensor
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor

model = load_to_cpu("../snapshots/sine_parallel")
print("model has " + str(model.parameter_count()) + " parameters")

model.output_length = 256

data = WavenetMixtureDataset(location='../_train_samples/sine',
                             item_length=model.receptive_field,
                             target_length=1)

start_data, _ = data[100]
start_data = start_data.unsqueeze(0)
generated = model.generate(num_samples=32000,
                           first_samples=start_data)
generated_np = generated.data.squeeze().numpy()
scipy.io.wavfile.write('../latest_generated_clip.wav', rate=16000, data=generated_np.astype(np.float32))