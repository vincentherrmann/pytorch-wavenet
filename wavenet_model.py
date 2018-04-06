import os
import os.path
import time
from wavenet_modules import *
from audio_data import *


class WaveNetModel(nn.Module):
    """
    A Complete Wavenet Model

    Args:
        layers (Int):               Number of layers in each block
        blocks (Int):               Number of wavenet blocks of this model
        dilation_channels (Int):    Number of channels for the dilated convolution
        residual_channels (Int):    Number of channels for the residual connection
        skip_channels (Int):        Number of channels for the skip connections
        classes (Int):              Number of possible values each sample can have
        output_length (Int):        Number of samples that are generated for each input
        kernel_size (Int):          Size of the dilation kernel
        dtype:                      Parameter type of this model

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`()`
        L should be the length of the receptive field
    """
    def __init__(self,
                 layers=10,
                 blocks=4,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=256,
                 end_channels=256,
                 classes=256,
                 output_length=32,
                 kernel_size=2,
                 dtype=torch.FloatTensor,
                 bias=False):

        super(WaveNetModel, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.classes = classes
        self.kernel_size = kernel_size
        self.dtype = dtype

        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilations = []
        self.dilated_queues = []
        # self.main_convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.classes,
                                    out_channels=residual_channels,
                                    kernel_size=1,
                                    bias=bias)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilations of this layer
                self.dilations.append((new_dilation, init_dilation))

                # dilated queues for fast generation
                self.dilated_queues.append(DilatedQueue(max_length=(kernel_size - 1) * new_dilation + 1,
                                                        num_channels=residual_channels,
                                                        dilation=new_dilation,
                                                        dtype=dtype))

                # dilated convolutions
                self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=kernel_size,
                                                   bias=bias))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=kernel_size,
                                                 bias=bias))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1,
                                                     bias=bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1,
                                                 bias=bias))

                receptive_field += additional_scope
                additional_scope *= 2
                init_dilation = new_dilation
                new_dilation *= 2

        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=1,
                                  bias=True)

        self.end_conv_2 = nn.Conv1d(in_channels=end_channels,
                                    out_channels=classes,
                                    kernel_size=1,
                                    bias=True)

        # self.output_length = 2 ** (layers - 1)
        self.output_length = output_length
        self.receptive_field = receptive_field

    def wavenet(self, input, dilation_func):

        x = self.start_conv(input)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            (dilation, init_dilation) = self.dilations[i]

            residual = dilation_func(x, dilation, init_dilation, i)

            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = F.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = F.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            if x.size(2) != 1:
                 s = dilate(x, 1, init_dilation=dilation)
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, -s.size(2):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, (self.kernel_size - 1):]

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x

    def wavenet_dilate(self, input, dilation, init_dilation, i):
        x = dilate(input, dilation, init_dilation)
        return x

    def queue_dilate(self, input, dilation, init_dilation, i):
        queue = self.dilated_queues[i]
        queue.enqueue(input.data[0])
        x = queue.dequeue(num_deq=self.kernel_size,
                          dilation=dilation)
        x = x.unsqueeze(0)

        return x

    def forward(self, input):
        x = self.wavenet(input,
                         dilation_func=self.wavenet_dilate)

        # reshape output
        [n, c, l] = x.size()
        l = self.output_length
        x = x[:, :, -l:]
        x = x.transpose(1, 2).contiguous()
        x = x.view(n * l, c)
        return x

    def generate(self,
                 num_samples,
                 first_samples=None,
                 temperature=1.):
        self.eval()
        if first_samples is None:
            first_samples = self.dtype(1).zero_()
        generated = Variable(first_samples, volatile=True)

        num_pad = self.receptive_field - generated.size(0)
        if num_pad > 0:
            generated = constant_pad_1d(generated, self.scope, pad_start=True)
            print("pad zero")

        for i in range(num_samples):
            input = Variable(torch.FloatTensor(1, self.classes, self.receptive_field).zero_())
            input = input.scatter_(1, generated[-self.receptive_field:].view(1, -1, self.receptive_field), 1.)

            x = self.wavenet(input,
                             dilation_func=self.wavenet_dilate)[:, :, -1].squeeze()

            if temperature > 0:
                x /= temperature
                prob = F.softmax(x, dim=0)
                prob = prob.cpu()
                np_prob = prob.data.numpy()
                x = np.random.choice(self.classes, p=np_prob)
                x = Variable(torch.LongTensor([x]))#np.array([x])
            else:
                x = torch.max(x, 0)[1].float()

            generated = torch.cat((generated, x), 0)

        generated = (generated / self.classes) * 2. - 1
        mu_gen = mu_law_expansion(generated, self.classes)

        self.train()
        return mu_gen

    def generate_fast(self,
                      num_samples,
                      first_samples=None,
                      temperature=1.,
                      regularize=0.,
                      progress_callback=None,
                      progress_interval=100):
        self.eval()
        if first_samples is None:
            first_samples = torch.LongTensor(1).zero_() + (self.classes // 2)
        first_samples = Variable(first_samples)

        # reset queues
        for queue in self.dilated_queues:
            queue.reset()

        num_given_samples = first_samples.size(0)
        total_samples = num_given_samples + num_samples

        input = Variable(torch.FloatTensor(1, self.classes, 1).zero_())
        input = input.scatter_(1, first_samples[0:1].view(1, -1, 1), 1.)

        # fill queues with given samples
        for i in range(num_given_samples - 1):
            x = self.wavenet(input,
                             dilation_func=self.queue_dilate)
            input.zero_()
            input = input.scatter_(1, first_samples[i + 1:i + 2].view(1, -1, 1), 1.).view(1, self.classes, 1)

            # progress feedback
            if i % progress_interval == 0:
                if progress_callback is not None:
                    progress_callback(i, total_samples)

        # generate new samples
        generated = np.array([])
        regularizer = torch.pow(Variable(torch.arange(self.classes)) - self.classes / 2., 2)
        regularizer = regularizer.squeeze() * regularize
        tic = time.time()
        for i in range(num_samples):
            x = self.wavenet(input,
                             dilation_func=self.queue_dilate).squeeze()

            x -= regularizer

            if temperature > 0:
                # sample from softmax distribution
                x /= temperature
                prob = F.softmax(x, dim=0)
                prob = prob.cpu()
                np_prob = prob.data.numpy()
                x = np.random.choice(self.classes, p=np_prob)
                x = np.array([x])
            else:
                # convert to sample value
                x = torch.max(x, 0)[1][0]
                x = x.cpu()
                x = x.data.numpy()

            o = (x / self.classes) * 2. - 1
            generated = np.append(generated, o)

            # set new input
            x = Variable(torch.from_numpy(x).type(torch.LongTensor))
            input.zero_()
            input = input.scatter_(1, x.view(1, -1, 1), 1.).view(1, self.classes, 1)

            if (i+1) == 100:
                toc = time.time()
                print("one generating step does take approximately " + str((toc - tic) * 0.01) + " seconds)")

            # progress feedback
            if (i + num_given_samples) % progress_interval == 0:
                if progress_callback is not None:
                    progress_callback(i + num_given_samples, total_samples)

        self.train()
        mu_gen = mu_law_expansion(generated, self.classes)
        return mu_gen


    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

    def cpu(self, type=torch.FloatTensor):
        self.dtype = type
        for q in self.dilated_queues:
            q.dtype = self.dtype
        super().cpu()


def load_latest_model_from(location, use_cuda=True):
    files = [location + "/" + f for f in os.listdir(location)]
    newest_file = max(files, key=os.path.getctime)
    print("load model " + newest_file)

    if use_cuda:
        model = torch.load(newest_file)
    else:
        model = load_to_cpu(newest_file)

    return model


def load_to_cpu(path):
    model = torch.load(path, map_location=lambda storage, loc: storage)
    model.cpu()
    return model
