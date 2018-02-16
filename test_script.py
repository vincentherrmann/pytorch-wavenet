import time
from wavenet_model import *
from audio_data import *
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
                     blocks=3,
                     dilation_channels=16,
                     residual_channels=16,
                     skip_channels=128,
                     end_channels=[128, 64],
                     classes=24,
                     output_length=8,
                     dtype=dtype)

data = WavenetMixtureDataset(location='_train_samples/alla_turca',
                             item_length=model.receptive_field + model.output_length - 1,
                             target_length=model.output_length)

# context_model = WaveNetModel(layers=6,
#                              blocks=2,
#                              dilation_channels=16,
#                              residual_channels=16,
#                              skip_channels=32,
#                              end_channels=32,
#                              classes=32,
#                              in_classes=256,
#                              dilation_factor=4,
#                              kernel_size=4,
#                              output_length=8,
#                              dtype=dtype)
#
# model = WaveNetModelWithContext(layers=10,
#                                 blocks=3,
#                                 dilation_channels=16,
#                                 residual_channels=16,
#                                 skip_channels=128,
#                                 end_channels=256,
#                                 classes=256,
#                                 output_length=16,
#                                 dtype=dtype,
#                                 context_stack=context_model)
#
# context_model.output_length = model.receptive_field + model.output_length - 1

#model = load_latest_model_from('snapshots')
#model = torch.load('snapshots/snapshot_2017-12-10_09-48-19')

# data = WavenetDatasetWithRandomConditioning(dataset_file='_train_samples/c_jam/conditioning_dataset.npz',
#                                             item_length=model.receptive_field + model.output_length - 1,
#                                             target_length=model.output_length,
#                                             file_location='_train_samples/c_jam',
#                                             test_stride=500,
#                                             conditioning_period=model.conditioning_period,
#                                             conditioning_breadth=5,
#                                             conditioning_channels=model.conditioning_channels[0])

# data = WavenetDatasetWithSineConditioning(dataset_file='_train_samples/alla_turca/conditioning_dataset.npz',
#                                           item_length=model.receptive_field + model.output_length - 1,
#                                           target_length=model.output_length,
#                                           file_location='_train_samples/alla_turca',
#                                           test_stride=500,
#                                           conditioning_period=model.conditioning_period,
#                                           conditioning_breadth=500,
#                                           conditioning_channels=model.conditioning_channels[0])


# data = WavenetDataset(dataset_file='_train_samples/saber/dataset.npz',
#                       item_length=model.receptive_field + model.output_length - 1,
#                       target_length=model.output_length,
#                       file_location='_train_samples/saber',
#                       test_stride=20)

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

logger = Logger(log_interval=1,
                validation_interval=10)

# dataloader = torch.utils.data.DataLoader(data,
#                                          batch_size=32,
#                                          shuffle=True,
#                                          num_workers=0,
#                                          pin_memory=False)
#
# batch_num = 0
# for batch in iter(dataloader):
#     batch_num += 1
#     if batch_num % 100 == 0:
#         print("loaded batch number: ", batch_num)
#
# data.train = True
# batch_num = 0
# for batch in iter(dataloader):
#     batch_num += 1
#     if batch_num % 100 == 0:
#         print("loaded batch number: ", batch_num)

# logger = TensorboardLogger(log_interval=200,
#                            validation_interval=200,
#                            generate_interval=500,
#                            generate_function=generate_and_log_samples,
#                            log_dir="logs")


trainer = WavenetTrainer(model=model,
                         dataset=data,
                         lr=0.003,
                         weight_decay=0,
                         logger=logger,
                         snapshot_path='snapshots',
                         snapshot_name='saber_model',
                         snapshot_interval=500,
                         process_batch=data.process_batch,
                         loss_fun=mixture_loss,
                         dtype=dtype,
                         ltype=dtype)


#model.generate_fast(1000)

print('start training...')
tic = time.time()
trainer.train(batch_size=8,
              epochs=20)
toc = time.time()
print('Training took {} seconds.'.format(toc - tic))
