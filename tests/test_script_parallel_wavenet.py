from wavenet_model import *
from audio_data import *
from wavenet_training import *
from model_logging import *

dtype = torch.FloatTensor
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor

teacher_model = WaveNetModel(layers=10,
                             blocks=3,
                             dilation_channels=8,
                             residual_channels=8,
                             skip_channels=32,
                             end_channels=[8, 16],
                             classes=24,
                             output_length=64,
                             dtype=dtype,
                             bias=True)

student_model = ParallelWaveNet(stacks=3,
                                layers=10,
                                blocks=1,
                                dilation_channels=8,
                                residual_channels=8,
                                skip_channels=32,
                                end_channels=[16],
                                output_length=64,
                                bias=True)

receptive_field = max(teacher_model.receptive_field, student_model.receptive_field)

data = WavenetMixtureDataset(location='../_train_samples/alla_turca',
                             item_length=receptive_field,
                             target_length=1)

logger = Logger(log_interval=1,
                validation_interval=100)

trainer = DistillationTrainer(student_model=student_model,
                              teacher_model=teacher_model,
                              dataset=data,
                              logger=logger)

trainer.train(batch_size=8, epochs=10)