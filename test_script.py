import time
from wavenet_model import *
from audio_data import WavenetDataset
from wavenet_training import *
from model_logging import *
from scipy.io import wavfile

dtype = torch.FloatTensor
ltype = torch.LongTensor

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor

model = WaveNetModel(layers=8,
                     blocks=4,
                     dilation_channels=16,
                     residual_channels=16,
                     skip_channels=16,
                     output_length=8,
                     dtype=dtype)

#model = load_latest_model_from('snapshots')
#model = torch.load('snapshots/snapshot_2017-12-10_09-48-19')

data = WavenetDataset(dataset_file='train_samples/saber/dataset.npz',
                      item_length=model.receptive_field + model.output_length - 1,
                      target_length=model.output_length,
                      file_location='train_samples/saber',
                      test_stride=20)

# torch.save(model, 'untrained_model')
print('the dataset has ' + str(len(data)) + ' items')
print('model: ', model)
print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())


def generate_and_log_samples(step):
    sample_length=4000
    gen_model = load_latest_model_from('snapshots')
    print("start generating...")
    samples = generate_audio(gen_model,
                             length=sample_length,
                             temperatures=[0])
    tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    logger.audio_summary('temperature 0', tf_samples, step, sr=16000)

    samples = generate_audio(gen_model,
                             length=sample_length,
                             temperatures=[0.5])
    tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    logger.audio_summary('temperature 0.5', tf_samples, step, sr=16000)
    print("audio clips generated")

logger = TensorboardLogger(log_interval=200,
                           validation_interval=200,
                           generate_interval=500,
                           generate_function=generate_and_log_samples,
                           log_dir="logs")

trainer = WavenetTrainer(model=model,
                           dataset=data,
                           lr=0.0001,
                           weight_decay=0.1,
                           logger=logger,
                           snapshot_path='snapshots',
                           snapshot_name='saber_model',
                           snapshot_interval=500)

print('start training...')
tic = time.time()
trainer.train(batch_size=8,
              epochs=20)
toc = time.time()
print('Training took {} seconds.'.format(toc - tic))
