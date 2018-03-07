from wavenet_model import *
from audio_data import *
from wavenet_training import *
from model_logging import *
from torch.autograd import Variable

dtype = torch.FloatTensor
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor


def dummy_teacher(input):
    size = input.size(0) * student_model.output_length
    d = torch.zeros(size, 3)
    d[:, 0] += 1.
    d[:, 1] += -0.5
    d[:, 2] += -6.
    d = Variable(d)
    return d


teacher_model = load_to_cpu("../snapshots/sine_mix_model")
teacher_model.output_length = 256

student_model = ParallelWaveNet(stacks=3,
                                layers=10,
                                blocks=1,
                                dilation_channels=16,
                                residual_channels=16,
                                skip_channels=64,
                                end_channels=[16],
                                output_length=teacher_model.input_length+1,
                                bias=True)
print("student receptive field: ", student_model.receptive_field)
#student_model = load_to_cpu("../snapshots/sine_parallel")

# receptive_field = max(teacher_model.receptive_field, student_model.receptive_field)
#
# data = WavenetMixtureDataset(location='../_train_samples/sine',
#                              item_length=receptive_field,
#                              target_length=1)

logger = Logger(log_interval=1,
                validation_interval=10)

trainer = DistillationTrainer(student_model=student_model,
                              teacher_model=teacher_model,
                              logger=logger)

trainer.train(batch_size=8, sample_count=128)