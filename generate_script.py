import librosa
from wavenet_model import *
from audio_data import *
from wavenet_training import *
import scipy.io.wavfile
import numpy as np

#model = load_latest_model_from('snapshots', use_cuda=False)
model = load_to_cpu("snapshots/bach_model_relu_1_62k")
model.sampling_function = sample_from_mixture
#model.output_channels = model.classes

print('model: ', model)
print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())

data = WavenetMixtureDatasetWithConditioning(location='_train_samples/bach_violin',
                             item_length=model.receptive_field + model.output_length - 1,
                             target_length=model.output_length,
                             conditioning_channels=model.conditioning_channels[0])

# data = WavenetDatasetWithRandomConditioning(dataset_file='_train_samples/alla_turca/conditioning_dataset.npz',
#                                             item_length=model.receptive_field + model.output_length - 1,
#                                             target_length=model.output_length,
#                                             file_location='_train_samples/alla_turca',
#                                             test_stride=4000,
#                                             conditioning_period=5000,#model.conditioning_period,
#                                             conditioning_breadth=5,
#                                             conditioning_channels=8)#model.conditioning_channels[0])

# data = WavenetDataset(dataset_file='_train_samples/bach_chaconne/dataset.npz',
#                       item_length=model.receptive_field + model.output_length - 1,
#                       target_length=model.output_length,
#                       file_location='train_samples/bach_chaconne',
#                       test_stride=20)
print('the dataset has ' + str(len(data)) + ' items')

# start_data = data[35000]
# start_data = data.process_batch(start_data, torch.FloatTensor, torch.LongTensor)
# start_data, conditioning, offset = start_data[0]
# start_data = torch.max(start_data.data, 0)[1].type(torch.FloatTensor)
#

start_data = data[5000]
#start_data = omit_conditioning(start_data, torch.FloatTensor, torch.LongTensor)
start_data = data.process_batch(start_data, torch.FloatTensor, torch.LongTensor)
conditioning = start_data[0][1]
file_encoding = start_data[0][2]
start_data = start_data[0][0].data[0,:]


def prog_callback(step, total_steps):
    print(str(100 * step // total_steps) + "% generated")


generated = model.generate_fast(num_samples=16000,
                                 first_samples=start_data,
                                 progress_callback=prog_callback,
                                 progress_interval=100,
                                 temperature=1.0,
                                 regularize=0.,
                                 conditioning=conditioning,
                                 file_encoding=file_encoding,
                                 sampling_function=sample_from_mixture)

print(generated)
scipy.io.wavfile.write('latest_generated_clip.wav', rate=16000, data=generated.astype(np.float32))
#librosa.output.write_wav('latest_generated_clip.aiff', generated, sr=16000)