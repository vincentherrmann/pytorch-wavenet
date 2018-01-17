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

context_model = WaveNetModel(layers=10,
                             blocks=2,
                             dilation_channels=16,
                             residual_channels=16,
                             skip_channels=32,
                             end_channels=32,
                             classes=32,
                             in_classes=256,
                             output_length=8,
                             bias=True,
                             dtype=dtype)

model = WaveNetModelWithContext(layers=10,
                                blocks=3,
                                dilation_channels=16,
                                residual_channels=16,
                                skip_channels=512,
                                end_channels=512,
                                classes=256,
                                output_length=16,
                                bias=True,
                                dtype=dtype,
                                context_stack=context_model)

#model = load_latest_model_from('snapshots', use_cuda=True)
#model = torch.load('snapshots/some_model')

if use_cuda:
    print("move model to gpu")
    context_model.cuda()
    model.cuda()

print('model: ', model)
print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())

data = WavenetDataset(dataset_file='train_samples/bach_chaconne/dataset.npz',
                      item_length=model.receptive_field + model.output_length - 1,
                      target_length=model.output_length,
                      file_location='train_samples/bach_chaconne',
                      test_stride=500)
print('the dataset has ' + str(len(data)) + ' items')


def generate_and_log_samples(step):
    sample_length=32000
    gen_model = load_latest_model_from('snapshots', use_cuda=False)
    print("start generating...")
    # samples = generate_audio(gen_model,
    #                          length=sample_length,
    #                          temperatures=[0.5])
    # tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    # logger.audio_summary('temperature_0.5', tf_samples, step, sr=16000)

    samples = generate_audio(gen_model,
                             length=sample_length,
                             temperatures=[1.])
    tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    logger.audio_summary('temperature_1.0', tf_samples, step, sr=16000)
    print("audio clips generated")


logger = TensorboardLogger(log_interval=200,
                           validation_interval=400,
                           generate_interval=800,
                           generate_function=generate_and_log_samples,
                           log_dir="logs/chaconne_model")

trainer = WavenetTrainer(model=model,
                         dataset=data,
                         lr=0.0001,
                         weight_decay=0.0,
                         snapshot_path='snapshots',
                         snapshot_name='chaconne_model_context',
                         snapshot_interval=1000,
                         logger=logger,
                         dtype=dtype,
                         ltype=ltype)

print('start training...')
trainer.train(batch_size=16,
              epochs=10,
              continue_training_at_step=0)