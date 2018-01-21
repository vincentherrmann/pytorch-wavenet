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
                 in_classes=None,
                 output_length=32,
                 kernel_size=2,
                 dilation_factor=2,
                 dtype=torch.FloatTensor,
                 bias=False):

        super(WaveNetModel, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.classes = classes
        if in_classes is None:
            self.in_classes = classes
        else:
            self.in_classes = in_classes
        self.kernel_size = kernel_size
        self.dilation_factor = dilation_factor
        self.dtype = dtype
        self.use_bias = bias

        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilations = []
        self.dilated_queues = {}
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.in_classes,
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
                queue_name = 'layer_' + str(len(self.dilated_queues))
                self.dilated_queues[queue_name] = DilatedQueue(max_length=(kernel_size - 1) * new_dilation + 1,
                                                               num_channels=residual_channels,
                                                               dilation=new_dilation,
                                                               dtype=dtype)

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
                additional_scope *= self.dilation_factor
                init_dilation = new_dilation
                new_dilation *= self.dilation_factor

        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=1,
                                  bias=True)

        self.end_conv_2 = nn.Conv1d(in_channels=end_channels,
                                    out_channels=self.classes,
                                    kernel_size=1,
                                    bias=True)

        # self.output_length = 2 ** (layers - 1)
        self.output_length = output_length
        self.receptive_field = receptive_field
        self.activation_unit_init()

    def wavenet(self, input, dilation_func, activation_input={'x': None}):
        x = self.start_conv(input)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #			 |----------------------------------------|     *residual*
            #            |                                        |
            # 			 |	  |-- conv -- tanh --|			      |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #				  |-- conv -- sigm --|     |
            #							              1x1
            #							               |
            # ---------------------------------------> + ------------->	*skip*

            (dilation, init_dilation) = self.dilations[i]

            residual = dilation_func(x, dilation, init_dilation, queue='layer_' + str(i))
            activation_input['x'] = residual

            # dilated convolution
            x = self.activation_unit(activation_input, i, dilation_func)

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

    def activation_unit_init(self):
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        for _ in range(len(self.skip_convs)):
            # dilated convolutions
            self.filter_convs.append(nn.Conv1d(in_channels=self.residual_channels,
                                               out_channels=self.dilation_channels,
                                               kernel_size=self.kernel_size,
                                               bias=self.use_bias))

            self.gate_convs.append(nn.Conv1d(in_channels=self.residual_channels,
                                             out_channels=self.dilation_channels,
                                             kernel_size=self.kernel_size,
                                             bias=self.use_bias))

    def activation_unit(self, input, layer_index, dilation_func):
        # gated activation unit
        filter = self.filter_convs[layer_index](input['x'])
        filter = F.tanh(filter)
        gate = self.gate_convs[layer_index](input['x'])
        gate = F.sigmoid(gate)
        x = filter * gate
        return x

    def wavenet_dilate(self, input, dilation, init_dilation, queue=''):
        x = dilate(input, dilation, init_dilation)
        return x

    def queue_dilate(self, input, dilation, init_dilation, queue=''):
        queue = self.dilated_queues[queue]
        queue.enqueue(input.data[0])
        x = queue.dequeue(num_deq=self.kernel_size,
                          dilation=dilation)
        x = x.unsqueeze(0)

        return x

    def forward(self, input):
        input = input[:, :, -(self.receptive_field + self.output_length - 1):]
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
            input = generated[-self.receptive_field:].view(1, 1, -1)
            x = self.wavenet(input,
                             dilation_func=self.wavenet_dilate)[-1, :].squeeze()

            if temperature > 0:
                x /= temperature
                prob = F.softmax(x, dim=0)
                prob = prob.cpu()
                np_prob = prob.data.numpy()
                x = np.random.choice(self.classes, p=np_prob)
                x = np.array([x])

                soft_o = F.softmax(x)
                soft_o = soft_o.cpu()
                np_o = soft_o.data.numpy()
                s = np.random.choice(self.classes, p=np_o)
                s = Variable(torch.FloatTensor([s]))
                s = (s / self.classes) * 2. - 1
            else:
                max = torch.max(x, 0)[1].float()
                s = (max / self.classes) * 2. - 1  # new sample

            generated = torch.cat((generated, s), 0)
        self.train()
        return generated.data.tolist()

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
            generated = np.append(generated, x)

            # set new input
            x = Variable(torch.from_numpy(x).type(torch.LongTensor))
            input.zero_()
            input = input.scatter_(1, x.view(1, -1, 1), 1.).view(1, self.classes, 1)

            if (i+1) == 100:
                toc = time.time()
                print("one generating step does take approximately " + str((toc - tic) * 0.01) + " seconds)")

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
        for _, q in self.dilated_queues.items():
            q.dtype = self.dtype
        super().cpu()


class WaveNetModelWithContext(WaveNetModel):
    def __init__(self, *args, **kwargs):
        try:
            context_stack = kwargs.pop('context_stack')
        except KeyError:
            context_stack = None
        super().__init__(*args, **kwargs)
        self.context_stack = context_stack

        self.context_queues = []
        self.filter_context_convs = nn.ModuleList()
        self.gate_context_convs = nn.ModuleList()

        max_dilation = max([d[0] for d in self.dilations])
        self.dilated_queues['context'] = DilatedQueue(max_length=max_dilation * (self.kernel_size-1),
                                                      num_channels=self.context_stack.classes,
                                                      dilation=1,
                                                      dtype=self.context_stack.dtype)

        for l in range(len(self.skip_convs)):
            self.filter_context_convs.append(nn.Conv1d(in_channels=self.context_stack.classes,
                                                       out_channels=self.dilation_channels,
                                                       kernel_size=1,
                                                       bias=True))

            self.gate_context_convs.append(nn.Conv1d(in_channels=self.context_stack.classes,
                                                     out_channels=self.dilation_channels,
                                                     kernel_size=1,
                                                     bias=True))

    def forward(self, input):
        context = self.context_stack(input)
        context = context.view(input.size(0), -1, self.context_stack.classes)
        context = context.transpose(1, 2).contiguous()

        activation_input = {'x': None, 'context': context}
        input = input[:, :, -(self.receptive_field + self.output_length - 1):]
        x = self.wavenet(input,
                         dilation_func=self.wavenet_dilate,
                         activation_input=activation_input)

        # reshape output
        [n, c, l] = x.size()
        l = self.output_length
        x = x[:, :, -l:]
        x = x.transpose(1, 2).contiguous()
        x = x.view(n * l, c)
        return x

    def activation_unit(self, input, layer_index, dilation_func):
        # gated activation unit with context
        filter = self.filter_convs[layer_index](input['x'])
        gate = self.gate_convs[layer_index](input['x'])

        context = input['context']
        dilation = input['x'].size(0) // context.size(0)
        context = dilation_func(context, dilation, init_dilation=1, queue='context')
        context = context[:, :, -filter.size(2):]

        filter_context = self.filter_context_convs[layer_index](context)
        gate_context = self.gate_context_convs[layer_index](context)

        filter = F.tanh(filter + filter_context)
        gate = F.sigmoid(gate + gate_context)

        x = filter * gate
        return x

    def generate_fast(self,
                      num_samples,
                      first_samples=None,
                      temperature=1.,
                      regularize=0.,
                      progress_callback=None,
                      progress_interval=100):
        self.context_stack.eval()
        self.eval()
        if first_samples is None:
            first_samples = torch.LongTensor(1).zero_() + (self.classes // 2)
        first_samples = Variable(first_samples)

        # reset queues
        for _, queue in self.context_stack.dilated_queues.items():
            queue.reset()
        for _, queue in self.dilated_queues.items():
            queue.reset()

        num_given_samples = first_samples.size(0)
        total_samples = num_given_samples + num_samples

        # create one hot input encoding
        input = Variable(torch.FloatTensor(1, self.classes, 1).zero_())
        input = input.scatter_(1, first_samples[0:1].view(1, -1, 1), 1.)

        # fill queues with given samples
        for i in range(num_given_samples - 1):
            context = self.context_stack.wavenet(input,
                                                 dilation_func=self.context_stack.queue_dilate)
            context = context.view(input.size(0), -1, self.context_stack.classes)
            context = context.transpose(1, 2).contiguous()

            activation_input = {'x': None, 'context': context}
            x = self.wavenet(input,
                             dilation_func=self.queue_dilate,
                             activation_input=activation_input)
            input.zero_()
            input = input.scatter_(1, first_samples[i + 1:i + 2].view(1, -1, 1), 1.).view(1, self.classes, 1)

            # progress feedback
            if i % progress_interval == 0:
                if progress_callback is not None:
                    progress_callback(i, total_samples)

        # generate new samples
        generated = np.array([])
        #regularizer = torch.pow(Variable(torch.arange(self.classes)) - self.classes / 2., 2)
        #regularizer = regularizer.squeeze() * regularize
        tic = time.time()
        for i in range(num_samples):
            context = self.context_stack.wavenet(input,
                                                 dilation_func=self.context_stack.queue_dilate)
            context = context.view(input.size(0), -1, self.context_stack.classes)
            context = context.transpose(1, 2).contiguous()

            activation_input = {'x': None, 'context': context}
            x = self.wavenet(input,
                             dilation_func=self.queue_dilate,
                             activation_input=activation_input).squeeze()

            #x -= regularizer

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
            generated = np.append(generated, x)

            # set new input
            x = Variable(torch.from_numpy(x).type(torch.LongTensor))
            input.zero_()
            input = input.scatter_(1, x.view(1, -1, 1), 1.).view(1, self.classes, 1)

            if (i + 1) == 100:
                toc = time.time()
                print("one generating step does take approximately " + str((toc - tic) * 0.01) + " seconds)")

            # progress feedback
            if (i + num_given_samples) % progress_interval == 0:
                if progress_callback is not None:
                    progress_callback(i + num_given_samples, total_samples)

        self.train()
        mu_gen = mu_law_expansion(generated, self.classes)
        return mu_gen

    def cpu(self, type=torch.FloatTensor):
        self.context_stack.cpu()
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
