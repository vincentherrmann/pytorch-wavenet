import time
from wavenet_model import *
from audio_data import WavenetDataset
from wavenet_training import WavenetTrainer
from model_logging import *
from scipy.io import wavfile

model = WaveNetModel(layers=6,
                     blocks=4,
                     dilation_channels=16,
                     residual_channels=16,
                     skip_channels=16,
                     output_length=8)

#model = load_latest_model_from('snapshots')
#model = torch.load('snapshots/snapshot_2017-12-10_09-48-19')

data = WavenetDataset(dataset_file='train_samples/clarinet/dataset.npz',
                      item_length=model.receptive_field + model.output_length - 1,
                      target_length=model.output_length,
                      file_location='train_samples/clarinet',
                      test_stride=20)

# torch.save(model, 'untrained_model')
print('the dataset has ' + str(len(data)) + ' items')
print('model: ', model)
print('receptive field: ', model.receptive_field)

logger = TensorboardLogger(log_interval=200,
                           validation_interval=200,
                           log_dir="logs")

trainer = WavenetTrainer(model=model,
                           dataset=data,
                           lr=0.0001,
                           weight_decay=0.1,
                           logger=logger,
                           snapshot_path='snapshots',
                           snapshot_name='clarinet_model',
                           snapshot_interval=500)

print('start training...')
tic = time.time()
trainer.train(batch_size=8,
              epochs=20)
toc = time.time()
print('Training took {} seconds.'.format(toc - tic))
