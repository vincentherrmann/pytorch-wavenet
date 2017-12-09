import torch
import torch.optim as optim
import torch.utils.data
import time
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from logger import Logger
from wavenet_modules import *


def print_last_loss(opt):
    print("loss: ", opt.losses[-1])


def print_last_validation_result(opt):
    print("validation loss: ", opt.validation_results[-1])


class WavenetOptimizer:
    def __init__(self,
                 model,
                 dataset,
                 optimizer=optim.Adam,
                 lr=0.001,
                 tensorboard_logger=None,
                 snapshot_path=None,
                 snapshot_interval=1000,
                 validate_interval=50):
        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.optimizer_type = optimizer
        self.optimizer = self.optimizer_type(params=self.model.parameters(), lr=self.lr)
        self.tensorboard_logger = tensorboard_logger
        self.snapshot_path = snapshot_path
        self.snapshot_interval = snapshot_interval
        self.validate_interval = validate_interval

    def train(self,
              batch_size=32,
              epochs=10):
        self.model.train()
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=8)
        step = 0
        for current_epoch in range(epochs):
            print("epoch", current_epoch)
            for (x, target) in iter(dataloader):
                x = Variable(x)
                target = Variable(target.view(-1))

                output = self.model(x)
                loss = F.cross_entropy(output.squeeze(), target.squeeze())
                loss.backward()
                loss = loss.data[0]
                self.optimizer.step()

                if self.tensorboard_logger is None:
                    if step % 10 == 0:
                        print("loss at step " + str(step) + ": " + str(loss))

                step += 1
                if step % self.snapshot_interval == 0:
                    if self.snapshot_path is None:
                        continue
                    time_string = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
                    torch.save(self.model, self.snapshot_path + '/snapshot_' + time_string)

                if step % self.validate_interval == 0:
                    self.validate(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        self.dataset.train = False
        total_loss = 0
        for (x, target) in iter(dataloader):
            x = Variable(x)
            target = Variable(target.view(-1))

            output = self.model(x)
            loss = F.cross_entropy(output.squeeze(), target.squeeze())
            total_loss += loss.data[0]
        print("validate model with " + str(len(dataloader.dataset)) + " samples")
        print("average loss: ", total_loss / len(dataloader))
        self.dataset.train = True
        self.model.train()


class WaveNetOptimizerOld:
    def __init__(self,
                 model,
                 data,
                 validation_segments=0,
                 examples_per_validation_segment=8,
                 optimizer=optim.Adam,
                 report_callback=print_last_loss,
                 report_interval=8,
                 validation_report_callback=print_last_validation_result,
                 logging_interval=64,
                 validation_interval=64,
                 snapshot_interval=256,
                 snapshot_file=None,
                 segments_per_chunk=16,
                 examples_per_segment=32):

        self.model = model
        self.data = data
        self.data.epoch_finished_callback = self.new_epoch
        self.learning_rate = 0.001
        self.optimizer_type = optimizer
        self.optimizer = optimizer(params=self.model.parameters(), lr=self.learning_rate)

        if validation_segments > 0:
            self.data.create_validation_set(segments=validation_segments,
                                            examples_per_segment=examples_per_validation_segment)

        self.report_callback = report_callback
        self.report_interval = report_interval
        self.validation_report_callback = validation_report_callback
        self.validation_interval = validation_interval
        self.logging_interval = logging_interval
        self.snapshot_interval = snapshot_interval
        self.snapshot_file = snapshot_file
        self.logger = Logger('./logs')


        self.i = 0 # current step
        self.losses = []
        self.step_times = []
        self.loss_positions = []
        self.validation_results = []
        self.validation_result_positions = []
        self.last_logged_validation = 0
        self.avg_loss = 0
        self.avg_time = 0
        self.current_epoch = -1
        self.epochs = 1
        self.segments_per_chunk = segments_per_chunk
        self.examples_per_segment = examples_per_segment

        self.new_epoch()
        self.data.load_new_chunk()
        self.data.use_new_chunk()

    def new_epoch(self):
        '''
        Start a new epoch or end training
        '''
        self.current_epoch += 1
        if self.current_epoch >= self.epochs:
            print("training finished")
            return

        print("epoch ", self.current_epoch)
        self.data.start_new_epoch(segments_per_chunk=self.segments_per_chunk,
                                  examples_per_segment=self.examples_per_segment)

    def validate_model(self, position, validation_m=16):
        '''
        Run model on validation set and report the result

        :param validation_m: number of examples from the validation set in one minibatch
        '''
        self.model.eval()
        avg_loss = 0
        i = 0

        while i < self.data.validation_index_count:
            inputs = self.data.validation_inputs[i:(i + validation_m), :, :]
            inputs = Variable(inputs, volatile=True)
            targets = self.data.validation_targets[i:(i + validation_m), :]
            targets = targets.view(targets.size(0) * targets.size(1))
            targets = Variable(targets, volatile=True)
            output = self.model(inputs)
            loss = F.cross_entropy(output.squeeze(), targets).data[0]
            avg_loss += loss
            i += validation_m

        avg_loss = avg_loss * validation_m / self.data.validation_index_count
        self.validation_results.append(avg_loss)
        self.validation_result_positions.append(position)
        if self.validation_report_callback is not None:
            self.validation_report_callback(self)

        self.model.train()

    def log_to_tensor_board(self):
        # TensorBoard logging

        # loss
        self.logger.scalar_summary("loss", self.avg_loss, self.i)

        # validation loss
        validation_position = self.validation_result_positions[-1]
        if validation_position > self.last_logged_validation:
            self.logger.scalar_summary("validation loss", self.validation_results[-1], validation_position)
            self.last_logged_validation = validation_position

        # parameter count
        self.logger.scalar_summary("parameter count", self.model.parameter_count(), self.i)

        # parameter histograms
        for tag, value, in self.model.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary(tag, value.data.cpu().numpy(), self.i)
            if value.grad is not None:
                self.logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), self.i)

        # normalized cross correlation
        for tag, module in self.model.named_modules():
            tag = tag.replace('.', '/')
            if type(module) is Conv1dExtendable:
                ncc = module.normalized_cross_correlation()
                self.logger.histo_summary(tag + '/ncc', ncc.data.cpu().numpy(), self.i)

    def log_normalized_cross_correlation(self):
        print("cross correlations")
        for name, module in self.model.named_modules():
            if type(module) is Conv1dExtendable:
                ncc = module.normalized_cross_correlation()
                print(ncc)

    def split_important_features(self, threshold):

        for name, module in self.model.named_modules():
            if module is self.model.end_conv:
                #print("Can't split feature in end conv")
                continue
            if type(module) is Conv1dExtendable:
                ncc = module.normalized_cross_correlation()

        splitted = False

        for name, module in self.model.named_modules():
            if module is self.model.end_conv:
                #print("Can't split feature in end conv")
                continue
            if type(module) is Conv1dExtendable:
                if len(module.output_tied_modules) > 0:
                    all_nccs = [module.current_ncc] + [m.current_ncc for m in module.output_tied_modules]
                    ncc_tensor = torch.abs(torch.stack(all_nccs))
                    ncc = torch.mean(ncc_tensor, dim=0)
                else:
                    ncc = module.current_ncc

                for feature_number, value in enumerate(ncc):
                    if abs(value.data[0]) > threshold:
                        print("in ", name, ", split feature number ", feature_number)
                        module.split_feature(feature_number=feature_number)
                        all_modules = [module] + module.output_tied_modules
                        [m.normalized_cross_correlation() for m in all_modules]
                        splitted = True

        if splitted:
            self.optimizer = self.optimizer_type(params=self.model.parameters(), lr=self.learning_rate)

    def reset_training(self):
        self.i = 0
        self.losses = []
        self.step_times = []
        self.loss_positions = []
        self.validation_results = []
        self.validation_result_positions = []
        self.avg_loss = 0
        self.avg_time = 0
        self.current_epoch = -1

        self.new_epoch()
        self.data.load_new_chunk()
        self.data.use_new_chunk()


    def train(self,
              learning_rate=0.001,
              minibatch_size=8,
              epochs=100,
              segments_per_chunk=16,
              examples_per_segment=32,
              split_threshold=0.2):

        '''
        Train a Wavenet model

        :param learning_rate: Learning rate of the optimizer
        :param minibatch_size: Number of examples in one minibatch
        :param epochs: Number of training epochs
        :param segments_per_chunk: Number of segments from the training data that are simultaneously loaded into memory
        :param examples_per_segment: The number of examples each of these segments contains
        '''

        self.learning_rate = learning_rate
        self.optimizer.lr = learning_rate
        self.epochs = epochs

        if segments_per_chunk != self.segments_per_chunk | examples_per_segment != self.examples_per_segment:
            self.segments_per_chunk = segments_per_chunk
            self.examples_per_segment = examples_per_segment

            self.new_epoch()
            self.data.load_new_chunk()
            self.data.use_new_chunk()

        self.model.train()  # set to train mode

        # train loop
        while True:
            tic = time.time()
            self.optimizer.zero_grad()

            # get data
            inputs, targets = self.data.get_minibatch(minibatch_size)
            targets = targets.view(targets.size(0) * targets.size(1))
            inputs = Variable(inputs)
            targets = Variable(targets)

            output = self.model(inputs)
            loss = F.cross_entropy(output.squeeze(), targets)
            loss.backward()

            loss = loss.data[0]
            # if loss > previous_loss * 3:
            # 	print("unexpected high loss: ", loss)
            # 	print("at minibatch ", minibatch_indices, " / ", data.data_length)

            self.optimizer.step()

            step_time = time.time() - tic
            self.avg_time += step_time
            self.avg_loss += loss
            self.i += 1

            # train feedback
            # if self.i % self.report_interval == 0:
                #print("loss: ", loss)
                #if self.report_callback != None:
                #    self.report_callback(self)

            # run on validation set
            if self.i % self.validation_interval == 0:
                self.validate_model(self.i)
                # print("average step time: ", self.step_times[-1])
                # print("validation loss: ", avg_loss)

            if self.i % self.logging_interval == 0:
                self.avg_loss /= self.logging_interval
                self.avg_time /= self.logging_interval
                previous_loss = self.avg_loss

                self.losses.append(self.avg_loss)
                self.step_times.append(self.avg_time)
                self.loss_positions.append(self.i)

                print("log to tensorBoard")
                self.log_to_tensor_board()
                self.split_important_features(threshold=split_threshold)
                #self.log_normalized_cross_correlation()

                self.avg_loss = 0
                self.avg_time = 0


            # snapshot
            if self.i % self.snapshot_interval == 0:
                if self.snapshot_file != None:
                    torch.save(self.model.state_dict(), self.snapshot_file)
                    date = str(datetime.now())
                    print(date, ": snapshot saved to ", self.snapshot_file)