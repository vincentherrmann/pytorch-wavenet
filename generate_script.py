import librosa
from wavenet_model import *
from audio_data import WavenetDataset
from wavenet_training import *

model = load_latest_model_from('snapshots', use_cuda=False)

print('model: ', model)
print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())

data = WavenetDataset(dataset_file='train_samples/bach_chaconne/dataset.npz',
                      item_length=model.receptive_field + model.output_length - 1,
                      target_length=model.output_length,
                      file_location='train_samples/bach_chaconne',
                      test_stride=20)
print('the dataset has ' + str(len(data)) + ' items')

start_data = data[250000][0]
start_data = torch.max(start_data, 0)[1]


def prog_callback(step, total_steps):
    print(str(100 * step // total_steps) + "% generated")


generated = model.generate_fast(num_samples=16000,
                                 first_samples=start_data,
                                 progress_callback=prog_callback,
                                 progress_interval=1000,
                                 temperature=1.0,
                                 regularize=0.)

print(generated)
librosa.output.write_wav('latest_generated_clip.wav', generated, sr=16000)