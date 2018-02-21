import torch
import torch.optim as optim
import torch.utils.data
import time
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from model_logging import Logger
from wavenet_modules import *


def print_last_loss(opt):
    print("loss: ", opt.losses[-1])


def print_last_validation_result(opt):
    print("validation loss: ", opt.validation_results[-1])

def mixture_loss(input, target, bin_count=256):
    loss = discretized_mix_logistic_loss(input, target, bin_count=bin_count, reduce=True)
    return loss


def mixture_accuracy(input, target, bin_count=256):
    modes = get_modes_from_discretized_mix_logistic(input, bin_count=bin_count)
    half_bin_size = 1./float(bin_count)
    accurate_predictions = torch.abs(target - modes) < half_bin_size
    accurate_prediction_count = torch.sum(accurate_predictions.int())
    return accurate_prediction_count.data[0]


def softmax_accuracy(input, target):
    predictions = torch.max(input, 1)[1].view(-1)
    correct_pred = torch.eq(target, predictions)
    correct_predictions = torch.sum(correct_pred).data[0]
    return correct_predictions


class WavenetTrainer:
    def __init__(self,
                 model,
                 dataset,
                 optimizer=optim.Adam,
                 lr=0.001,
                 weight_decay=0,
                 gradient_clipping=None,
                 logger=Logger(),
                 snapshot_path=None,
                 snapshot_name='snapshot',
                 snapshot_interval=1000,
                 snapshot_callback=None,
                 dtype=torch.FloatTensor,
                 ltype=torch.LongTensor,
                 num_workers=8,
                 pin_memory=False,
                 process_batch=None,
                 loss_fun=F.cross_entropy,
                 accuracy_fun=softmax_accuracy):
        self.model = model
        self.dataset = dataset
        self.dataloader = None
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip = gradient_clipping
        self.optimizer_type = optimizer
        self.optimizer = self.optimizer_type(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.logger = logger
        self.logger.trainer = self
        self.snapshot_path = snapshot_path
        self.snapshot_name = snapshot_name
        self.snapshot_interval = snapshot_interval
        self.snapshot_callback = snapshot_callback
        self.dtype = dtype
        self.ltype = ltype
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.process_batch = process_batch
        self.loss_fun = loss_fun
        self.accuracy_fun = accuracy_fun

    def train(self,
              batch_size=32,
              epochs=10,
              continue_training_at_step=0):
        self.model.train()
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=self.num_workers,
                                                      pin_memory=self.pin_memory)
        step = continue_training_at_step
        num_step_track = 10
        step_times = np.zeros(num_step_track)
        for current_epoch in range(epochs):
            print("epoch", current_epoch)
            tic = time.time()
            for batch in iter(self.dataloader):
                if self.process_batch is None:
                    x, target = batch
                    x = Variable(x.type(self.dtype))
                    target = Variable(target.type(self.ltype))
                else:
                    x, target = self.process_batch(batch, self.dtype, self.ltype)
                output = self.model(x)
                target = target.view(-1)
                loss = self.loss_fun(output.squeeze(), target.squeeze())
                self.optimizer.zero_grad()
                loss.backward()
                loss = loss.data[0]

                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
                self.optimizer.step()
                step += 1

                # time step duration:
                if step <= num_step_track:
                    toc = time.time()
                    step_times[step-1] = toc - tic
                    tic = time.time()
                    if step == num_step_track:
                        mean = np.mean(step_times)
                        std = np.std(step_times)
                        print("one training step does take " + str(mean) + " +/- " + str(std) + " seconds")

                if step % self.snapshot_interval == 0:
                    if self.snapshot_path is None:
                        continue
                    time_string = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
                    torch.save(self.model, self.snapshot_path + '/' + self.snapshot_name + '_' + time_string)
                    if self.snapshot_callback is not None:
                        self.snapshot_callback()

                self.logger.log(step, loss)

    def validate(self):
        self.model.eval()
        dataset_state = self.dataset.train  # remember the current state
        self.dataset.train = False
        total_loss = 0
        accurate_classifications = 0
        for batch in iter(self.dataloader):
            if self.process_batch is None:
                x, target = batch
                x = Variable(x.type(self.dtype))
                target = Variable(target.type(self.ltype))
            else:
                x, target = self.process_batch(batch, self.dtype, self.ltype)

            output = self.model(x)
            target = target.view(-1)
            loss = self.loss_fun(output.squeeze(), target.squeeze())
            total_loss += loss.data[0]

            correct_predictions = self.accuracy_fun(output, target)
            accurate_classifications += correct_predictions
        # print("validate model with " + str(len(self.dataloader.dataset)) + " samples")
        # print("average loss: ", total_loss / len(self.dataloader))
        avg_loss = total_loss / len(self.dataloader)
        avg_accuracy = accurate_classifications / (len(self.dataset)*self.dataset.target_length)
        self.dataset.train = dataset_state
        self.model.train()
        return avg_loss, avg_accuracy


class DistillationTrainer:
    def __init__(self,
                 student_model,
                 teacher_model,
                 dataset,
                 optimizer=optim.Adam,
                 lr=0.001,
                 pin_memory=False,
                 num_workers=0,
                 process_batch=None,
                 dtype=torch.FloatTensor,
                 ltype=torch.FloatTensor,
                 logger=Logger()):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.dataset = dataset
        self.dataloader = None
        self.lr = lr
        self.optimizer_type = optimizer
        self.optimizer = self.optimizer_type(params=self.student_model.parameters(), lr=self.lr)
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.process_batch = process_batch
        self.dtype = dtype
        self.ltype = ltype
        self.logger = logger

    def train(self,
              batch_size=32,
              epochs=10,
              continue_training_at_step=0,
              sample_count=16):
        self.student_model.train()
        self.teacher_model.eval()
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=self.num_workers,
                                                      pin_memory=self.pin_memory)
        step = continue_training_at_step
        for current_epoch in range(epochs):
            print("epoch", current_epoch)
            tic = time.time()
            for batch in iter(self.dataloader):
                if self.process_batch is None:
                    x, target = batch
                    x = Variable(x.type(self.dtype))
                    target = Variable(target.type(self.ltype))
                else:
                    x, target = self.process_batch(batch, self.dtype, self.ltype)

                u = torch.FloatTensor(batch_size, 1, self.student_model.output_length)
                if x.is_cuda:
                    u = u.cuda()
                u.uniform_(1e-5, 1. - 1e-5)
                u = Variable(u, requires_grad=False)
                z = torch.log(u) - torch.log(1. - u)
                output, mu, s = self.student_model(x, z)

                teacher_input = torch.cat([x, output], dim=2)
                target_distribution = self.teacher_model(teacher_input)
                entropy = torch.sum(s.view(-1))

                # sample from student distribution
                [n, _, l] = output.size()
                u = torch.FloatTensor(n*l, sample_count)
                if output.is_cuda:
                    u = u.cuda()
                u.uniform_(1e-5, 1. - 1e-5)
                u = Variable(u, requires_grad=False)
                student_samples = mu.view(-1, 1) + torch.exp(s).view(-1, 1) * (torch.log(u) - torch.log(1. - u))  # (n*l, s_c)
                cross_entropy = discretized_mix_logistic_loss(target_distribution, student_samples)
                loss = cross_entropy - entropy
                self.optimizer.zero_grad()
                loss.backward()
                loss = loss.data[0]

                self.optimizer.step()
                step += 1

                self.logger.log(step, loss)


def generate_audio(model,
                   length=8000,
                   temperatures=[0., 1.]):
    samples = []
    for temp in temperatures:
        samples.append(model.generate_fast(length, temperature=temp))
    samples = np.stack(samples, axis=0)
    return samples

