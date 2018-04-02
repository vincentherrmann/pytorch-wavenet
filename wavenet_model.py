import os
import os.path
import time
from wavenet_modules import *
from audio_data import *

wavenet_default_settings = {"layers": 10,
                            "blocks": 4,
                            "dilation_channels": 32,
                            "residual_channels": 32,
                            "skip_channels": 512,
                            "end_channels": [256, 128],
                            "output_channels": 24,
                            "output_length": 1024,
                            "kernel_size": 2,
                            "dilation_factor": 2,
                            "bias": True,
                            "dtype": torch.FloatTensor}

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

    Input should be mu-law encoded and between -1. and 1.
    """
    def __init__(self, args_dict=wavenet_default_settings):

        super(WaveNetModel, self).__init__()

        self.layers = args_dict["layers"]
        self.blocks = args_dict["blocks"]
        self.dilation_channels = args_dict["dilation_channels"]
        self.residual_channels = args_dict["residual_channels"]
        self.skip_channels = args_dict["skip_channels"]
        self.end_channels = args_dict["end_channels"]
        self.output_channels = args_dict["output_channels"]
        self.output_length = args_dict["output_length"]
        self.kernel_size = args_dict["kernel_size"]
        self.dilation_factor = args_dict["dilation_factor"]
        self.dtype = args_dict["dtype"]
        self.use_bias = args_dict["bias"]

        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilations = []
        self.dilated_queues = {}
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.end_layers = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=1, #self.in_classes,
                                    out_channels=self.residual_channels,
                                    kernel_size=1,
                                    bias=self.use_bias)

        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilations of this layer
                self.dilations.append((new_dilation, init_dilation))

                # dilated queues for fast generation
                queue_name = 'layer_' + str(len(self.dilated_queues))
                self.dilated_queues[queue_name] = DilatedQueue(max_length=(self.kernel_size - 1) * new_dilation + 1,
                                                               num_channels=self.residual_channels,
                                                               dilation=new_dilation,
                                                               dtype=self.dtype)

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=1,
                                                     bias=self.use_bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=1,
                                                 bias=self.use_bias))

                receptive_field += additional_scope
                additional_scope *= self.dilation_factor
                init_dilation = new_dilation
                new_dilation *= self.dilation_factor

        in_channels = self.skip_channels
        for end_channel in self.end_channels:
            self.end_layers.append(nn.Conv1d(in_channels=in_channels,
                                             out_channels=end_channel,
                                             kernel_size=1,
                                             bias=True))
            in_channels = end_channel

        self.end_layers.append(nn.Conv1d(in_channels=in_channels,
                                         out_channels=self.output_channels,
                                         kernel_size=1,
                                         bias=True))

        # self.output_length = 2 ** (layers - 1)
        self.receptive_field = receptive_field
        self.activation_unit_init()

    @property
    def input_length(self):
        return self.receptive_field + self.output_length - 1

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
            s = x.clone()
            if x.size(2) != 1:
                pass
            if x.size(2) != 0:  # 1: TODO: delete this line !? (why is it there?)
                 s = dilate(x, 1, init_dilation=dilation)
            s = self.skip_convs[i](s)
            # print("x size: ", x.size())
            # print("s size after skip_conv: ", s.size())
            try:
                skip = skip[:, :, -s.size(2):]
            except:
                skip = 0
            skip = s + skip

            del s

            x = self.residual_convs[i](x)
            x = x + residual[:, :, (self.kernel_size - 1):]

            del residual

        x = skip
        for this_layer in self.end_layers:
            x = this_layer(F.relu(x, inplace=True))

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
                x = np.random.choice(self.output_channels, p=np_prob)
                x = np.array([x])

                soft_o = F.softmax(x)
                soft_o = soft_o.cpu()
                np_o = soft_o.data.numpy()
                s = np.random.choice(self.output_channels, p=np_o)
                s = Variable(torch.FloatTensor([s]))
                s = (s / self.output_channels) * 2. - 1
            else:
                max = torch.max(x, 0)[1].float()
                s = (max / self.output_channels) * 2. - 1  # new sample

            generated = torch.cat((generated, s), 0)
        self.train()
        return generated.data.tolist()

    def generate_fast(self,
                      num_samples,
                      first_samples=None,
                      temperature=1.,
                      regularize=0.,
                      progress_callback=None,
                      progress_interval=100,
                      sampling_function=sample_from_softmax):
        self.eval()
        if first_samples is None:
            first_samples = self.dtype(1).zero_()
            #ONE HOT: first_samples = torch.LongTensor(1).zero_() + (self.classes // 2)
        elif first_samples.size[0] > self.receptive_field:
            first_samples = first_samples[-self.receptive_field:]
        first_samples = Variable(first_samples, volatile=True)

        # reset queues
        for _, queue in self.dilated_queues.items():
            queue.reset()

        num_given_samples = first_samples.size(0)
        total_samples = num_given_samples + num_samples

        input = first_samples[0:1].view(1, 1, 1)
        # ONE HOT
        # input = Variable(torch.FloatTensor(1, self.classes, 1).zero_(), volatile=True)
        # input = input.scatter_(1, first_samples[0:1].view(1, -1, 1), 1.)

        # fill queues with given samples
        for i in range(num_given_samples - 1):
            x = self.wavenet(input,
                             dilation_func=self.queue_dilate)
            input = first_samples[i+1:i+2].view(1, 1, 1)
            # ONE HOT
            # input.zero_()
            # input = input.scatter_(1, first_samples[i + 1:i + 2].view(1, -1, 1), 1.).view(1, self.classes, 1)

            # progress feedback
            if i % progress_interval == 0:
                if progress_callback is not None:
                    progress_callback(i, total_samples)

        # generate new samples
        generated = first_samples.data.squeeze().numpy()  #np.array([])
        # regularizer = torch.pow(Variable(torch.arange(self.classes)) - self.classes / 2., 2)
        # regularizer = regularizer.squeeze() * regularize
        tic = time.time()
        for i in range(num_samples):
            x = self.wavenet(input,
                             dilation_func=self.queue_dilate).squeeze()

            # x -= regularizer

            if temperature > 0:
                o = sampling_function(x, temperature, self.output_channels)
                o = o.astype(np.float32)
            else:
                # convert to sample value
                x = torch.max(x, 0)[1][0]
                x = x.cpu()
                x = x.data.numpy()

            generated = np.append(generated, o)

            # set new input
            input = Variable(torch.from_numpy(o), volatile=True).view(1, 1, 1)
            # TODO: check cuda
            # ONE HOT
            # x = Variable(torch.from_numpy(x).type(torch.LongTensor))
            # input.zero_()
            # input = input.scatter_(1, x.view(1, -1, 1), 1.).view(1, self.classes, 1)

            if (i+1) == 100:
                toc = time.time()
                print("one generating step does take approximately " + str((toc - tic) * 0.01) + " seconds)")

            # progress feedback
            if (i + num_given_samples) % progress_interval == 0:
                if progress_callback is not None:
                    progress_callback(i + num_given_samples, total_samples)

        self.train()
        mu_gen = mu_law_expansion(generated, self.output_channels)
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


conditioning_wavenet_default_settings = wavenet_default_settings
conditioning_wavenet_default_settings["conditioning_channels"] = [16, 32, 16]
conditioning_wavenet_default_settings["file_encoding_channels"] = [32, 16]
conditioning_wavenet_default_settings["conditioning_period"] = 128
conditioning_wavenet_default_settings["conditioning_noise"] = 0.05


class WaveNetModelWithConditioning(WaveNetModel):
    def __init__(self, args_dict):
        self.conditioning_channels = args_dict["conditioning_channels"]
        self.file_encoding_channels = args_dict["file_encoding_channels"]
        self.conditioning_period = args_dict["conditioning_period"]
        self.conditioning_noise = args_dict["conditioning_noise"]

        super().__init__(args_dict)

        self.file_encoding_layers = nn.ModuleList()
        self.file_encoding_dropout = nn.Dropout(p=0.5)
        for i in range(len(self.file_encoding_channels) - 1):
            self.file_encoding_layers.append(nn.Conv1d(in_channels=self.file_encoding_channels[i],
                                                       out_channels=self.file_encoding_channels[i + 1],
                                                       kernel_size=1,
                                                       bias=self.use_bias))
        self.conditioning_layers = nn.ModuleList()
        self.conditioning_dropout = nn.Dropout(p=0.5)
        self.file_conditioning_cross_layers = nn.ModuleList()
        for i in range(len(self.conditioning_channels)-1):
            self.conditioning_layers.append(nn.Conv1d(in_channels=self.conditioning_channels[i],
                                                      out_channels=self.conditioning_channels[i+1],
                                                      kernel_size=1,
                                                      bias=False if i == 0 else self.use_bias))

            self.file_conditioning_cross_layers.append(nn.Conv1d(in_channels=self.file_encoding_channels[-1],
                                                                 out_channels=self.conditioning_channels[i+1],
                                                                 kernel_size=1,
                                                                 bias=self.use_bias))

    def activation_unit_init(self):
        super().activation_unit_init()

        self.filter_conditioning_convs = nn.ModuleList()
        self.gate_conditioning_convs = nn.ModuleList()
        for l in range(len(self.skip_convs)):
            self.filter_conditioning_convs.append(nn.Conv1d(in_channels=self.conditioning_channels[-1],
                                                            out_channels=self.dilation_channels,
                                                            kernel_size=1,
                                                            bias=self.use_bias))

            self.gate_conditioning_convs.append(nn.Conv1d(in_channels=self.conditioning_channels[-1],
                                                          out_channels=self.dilation_channels,
                                                          kernel_size=1,
                                                          bias=self.use_bias))

            this_dilation = self.dilations[l][0]
            queue_name = 'filter_conditioning_' + str(l)
            self.dilated_queues[queue_name] = DilatedQueue(max_length=(self.kernel_size - 1) * this_dilation + 1,
                                                           num_channels=self.residual_channels,
                                                           dilation=this_dilation,
                                                           dtype=self.dtype)
            queue_name = 'gate_conditioning_' + str(l)
            self.dilated_queues[queue_name] = DilatedQueue(max_length=(self.kernel_size - 1) * this_dilation + 1,
                                                           num_channels=self.residual_channels,
                                                           dilation=this_dilation,
                                                           dtype=self.dtype)

    def forward(self, input):
        input, conditioning, file_encoding, offset = input
        conditioning = self.conditional_network(conditioning, file_encoding)
        # for l in range(len(self.conditioning_layers)):
        #     if l != len(self.conditioning_layers) - 1:
        #         conditioning = F.relu(conditioning)
        #     conditioning = self.conditioning_layers[l](conditioning)

        activation_input = {'x': None, 'conditioning': conditioning, 'offset': offset}
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
        # gated activation unit with conditioning
        filter = self.filter_convs[layer_index](input['x'])
        gate = self.gate_convs[layer_index](input['x'])

        conditioning = input['conditioning']
        offset = input['offset']
        dilation = input['x'].size(0) // conditioning.size(0)

        filter_cond = self.filter_conditioning_convs[layer_index](conditioning)#.unsqueeze(3)
        gate_cond = self.gate_conditioning_convs[layer_index](conditioning)#.unsqueeze(3)

        # upsample conditioning by repeating the values (could also be done with a transposed convolution)
        n, c, l = filter_cond.shape
        filter_cond_rep = filter_cond.repeat(1, 1, 1, self.conditioning_period).view(n, c, -1)
        gate_cond_rep = gate_cond.repeat(1, 1, 1, self.conditioning_period).view(n, c, -1)

        l = self.receptive_field + self.output_length - 1
        filter_conditioning = torch.cat([filter_cond_rep[i:i+1, :, o:o+l] for i, o in enumerate(offset)])
        gate_conditioning = torch.cat([gate_cond_rep[i:i+1, :, o:o+l] for i, o in enumerate(offset)])
        filter_cond_dilated = dilation_func(filter_conditioning, dilation, init_dilation=1,
                                            queue='filter_conditioning_' + str(layer_index))
        gate_cond_dilated = dilation_func(gate_conditioning, dilation, init_dilation=1,
                                          queue='gate_conditioning_' + str(layer_index))
        l = filter.size(2)
        filter_conditioning = filter_cond_dilated[:, :, :l]
        gate_conditioning = gate_cond_dilated[:, :, :l]

        filter = F.tanh(filter + filter_conditioning)
        gate = F.sigmoid(gate + gate_conditioning)

        x = filter * gate
        return x

    def generate_fast(self,
                      num_samples,
                      first_samples=None,
                      temperature=1.,
                      regularize=0.,
                      progress_callback=None,
                      progress_interval=100,
                      conditioning=None,
                      file_encoding=None,
                      offset=0,
                      sampling_function=sample_from_softmax):
        self.eval()

        if first_samples is None:
            first_samples = self.dtype(1).zero_()
        if first_samples.size(0) > self.receptive_field:
            first_samples = first_samples[:self.receptive_field]
        first_samples = Variable(first_samples, volatile=True)

        # reset queues
        for _, queue in self.dilated_queues.items():
            queue.reset()

        num_given_samples = first_samples.size(0)
        total_samples = num_given_samples + num_samples
        conditioning_length = total_samples // self.conditioning_period + 2

        if conditioning is None:
            conditioning = Variable(self.dtype(self.conditioning_channels[0], conditioning_length).zero_())
        elif conditioning.size(1) < conditioning_length:
            # repeat the last conditioning
            remaining_length = conditioning_length - conditioning.size(1)
            end = conditioning[:, -1].contiguous().view(-1, 1).repeat(1, remaining_length)
            conditioning = torch.cat((conditioning, end), dim=1)

        if file_encoding is None:
            file_encoding = Variable(self.dtype(self.file_encoding_channels[0], conditioning_length).zero_())
        elif file_encoding.size(1) < conditioning_length:
            # repeat the last file encoding
            remaining_length = conditioning_length - file_encoding.size(1)
            end = file_encoding[:, -1].contiguous().view(-1, 1).repeat(1, remaining_length)
            file_encoding = torch.cat((file_encoding, end), dim=1)

        input = first_samples[0:1].view(1, 1, 1)

        original_conditioning_period = self.conditioning_period
        # set conditional period to 1 for fast generation, because the values are created sample for sample
        self.conditioning_period = 1

        # fill queues with given samples
        for i in range(num_given_samples - 1):
            cond_index = (i + offset) // original_conditioning_period
            cond = conditioning[:, cond_index].contiguous().view(1, -1, 1)
            enc = file_encoding[:, cond_index].contiguous().view(1, -1, 1)
            cond = self.conditional_network(cond, enc)
            activation_input = {'x': None, 'conditioning': cond, 'offset': [0]}
            _ = self.wavenet(input,
                             dilation_func=self.queue_dilate,
                             activation_input=activation_input)
            input = first_samples[i+1:i+2].view(1, 1, 1)
            # ONE HOT
            # input.zero_()
            # input = input.scatter_(1, first_samples[i + 1:i + 2].view(1, -1, 1), 1.).view(1, self.classes, 1)

            # progress feedback
            if i % progress_interval == 0:
                if progress_callback is not None:
                    progress_callback(i, total_samples)

        # generate new samples
        generated = first_samples.data.numpy() #np.array([])
        tic = time.time()
        for i in range(num_samples):
            cond_index = (num_given_samples + i + offset) // original_conditioning_period
            cond = conditioning[:, cond_index].contiguous().view(1, -1, 1)
            enc = file_encoding[:, cond_index].contiguous().view(1, -1, 1)
            cond = self.conditional_network(cond, enc)
            activation_input = {'x': None, 'conditioning': cond, 'offset': [0]}
            x = self.wavenet(input,
                             dilation_func=self.queue_dilate,
                             activation_input=activation_input).squeeze()

            if temperature > 0:
                o = sampling_function(x, temperature, self.output_channels)
                o = o.astype(np.float32)
            else:
                # convert to sample value
                x = torch.max(x, 0)[1][0]
                x = x.cpu()
                x = x.data.numpy()

            generated = np.append(generated, o)

            # set new input
            input = Variable(torch.from_numpy(o), volatile=True).view(1, 1, 1)

            if (i+1) == 100:
                toc = time.time()
                print("one generating step does take approximately " + str((toc - tic) * 0.01) + " seconds)")

            # progress feedback
            if (i + num_given_samples) % progress_interval == 0:
                if progress_callback is not None:
                    progress_callback(i + num_given_samples, total_samples)

        self.train()
        self.conditioning_period = original_conditioning_period
        #mu_gen = mu_law_expansion(generated, self.output_channels)
        #return mu_gen
        return generated

    def conditional_network(self, conditioning, file_encoding):
        if self.training:
            conditioning += Variable(torch.randn(conditioning.size()).type(self.dtype) * self.conditioning_noise)
        # TODO: This implementation is a bit clumsy..., reorder activation, dropout etc.
        for l in range(len(self.file_encoding_layers)):
            if l != 0:
                file_encoding = F.elu(file_encoding, inplace=True)
            file_encoding = self.file_encoding_layers[l](file_encoding)
        file_encoding = F.elu(file_encoding, inplace=True)
        file_encoding = self.file_encoding_dropout(file_encoding)

        for l in range(len(self.conditioning_layers)):
            if l != 0:
                cross_encoding = self.file_conditioning_cross_layers[l-1](file_encoding)
                conditioning = F.elu(conditioning + cross_encoding, inplace=True)
            conditioning = self.conditioning_layers[l](conditioning)
        cross_encoding = self.file_conditioning_cross_layers[-1](file_encoding)
        conditioning = F.elu(conditioning + cross_encoding, inplace=True)
        conditioning = self.conditioning_dropout(conditioning)
        return F.elu(conditioning, inplace=True)


class WaveNetModelReluWithConditioning(WaveNetModelWithConditioning):
    def activation_unit_init(self):
        self.activation_convs = nn.ModuleList()

        for _ in range(len(self.skip_convs)):
            self.activation_convs.append(nn.Conv1d(in_channels=self.residual_channels,
                                                   out_channels=self.dilation_channels,
                                                   kernel_size=self.kernel_size,
                                                   bias=self.use_bias))

        self.activation_conditioning_convs = nn.ModuleList()
        for l in range(len(self.skip_convs)):
            self.activation_conditioning_convs.append(nn.Conv1d(in_channels=self.conditioning_channels[-1],
                                                                out_channels=self.dilation_channels,
                                                                kernel_size=1,
                                                                bias=True))
            this_dilation = self.dilations[l][0]
            queue_name = 'conditioning_' + str(l)
            self.dilated_queues[queue_name] = DilatedQueue(max_length=(self.kernel_size - 1) * this_dilation + 1,
                                                           num_channels=self.residual_channels,
                                                           dilation=this_dilation,
                                                           dtype=self.dtype)

    def activation_unit(self, input, layer_index, dilation_func):
        # relu activation unit with conditioning
        preactivation = self.activation_convs[layer_index](input['x'])

        conditioning = input['conditioning']
        offset = input['offset']
        dilation = input['x'].size(0) // conditioning.size(0)

        conditioning = self.activation_conditioning_convs[layer_index](conditioning)

        # upsample conditioning by repeating the values (could also be done with a transposed convolution)
        n, c, l = conditioning.shape
        conditioning = conditioning.repeat(1, 1, 1, self.conditioning_period).view(n, c, -1)

        l = self.receptive_field + self.output_length - 1
        conditioning = torch.cat([conditioning[i:i + 1, :, o:o + l] for i, o in enumerate(offset)])
        conditioning = dilation_func(conditioning, dilation, init_dilation=1,
                                                   queue='conditioning_' + str(layer_index))

        l = preactivation.size(2)
        conditioning = conditioning[:, :, :l]

        x = F.relu(preactivation + conditioning, inplace=True)
        return x

    # def activation_unit(self, input, layer_index, dilation_func):
    #     # relu activation unit with conditioning
    #     preactivation = self.activation_convs[layer_index](input['x'])
    #
    #     conditioning = input['conditioning']
    #     offset = input['offset']
    #     dilation = input['x'].size(0) // conditioning.size(0)
    #
    #     preactivation_cond = self.activation_conditioning_convs[layer_index](conditioning)
    #
    #     # upsample conditioning by repeating the values (could also be done with a transposed convolution)
    #     n, c, l = preactivation_cond.shape
    #     preactivation_cond_rep = preactivation_cond.repeat(1, 1, 1, self.conditioning_period).view(n, c, -1)
    #     gate_cond_rep = preactivation_cond.repeat(1, 1, 1, self.conditioning_period).view(n, c, -1)
    #
    #     l = self.receptive_field + self.output_length - 1
    #     preactivation_conditioning = torch.cat([preactivation_cond_rep[i:i + 1, :, o:o + l] for i, o in enumerate(offset)])
    #     preactivation_cond_dilated = dilation_func(preactivation_conditioning, dilation, init_dilation=1,
    #                                                queue='conditioning_' + str(layer_index))
    #
    #     l = preactivation.size(2)
    #     preactivation_conditioning = preactivation_cond_dilated[:, :, :l]
    #
    #     x = F.relu(preactivation + preactivation_conditioning, inplace=True)
    #     return x


class WaveNetModelWithContext(WaveNetModel):
    def __init__(self, *args, **kwargs):
        try:
            context_stack = kwargs.pop('context_stack')
        except KeyError:
            context_stack = None
        super().__init__(*args, **kwargs)
        self.context_stack = context_stack

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
            first_samples = self.dtype(1).zero_()
            #ONE HOT: first_samples = torch.LongTensor(1).zero_() + (self.classes // 2)
        first_samples = Variable(first_samples, volatile=True)

        # reset queues
        for _, queue in self.context_stack.dilated_queues.items():
            queue.reset()
        for _, queue in self.dilated_queues.items():
            queue.reset()

        num_given_samples = first_samples.size(0)
        total_samples = num_given_samples + num_samples

        input = first_samples[0:1].view(1, 1, 1)
        # create one hot input encoding
        # input = Variable(torch.FloatTensor(1, self.classes, 1).zero_())
        # input = input.scatter_(1, first_samples[0:1].view(1, -1, 1), 1.)

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
            input = first_samples[i + 1:i + 2].view(1, 1, 1)
            # ONE HOT
            # input.zero_()
            # input = input.scatter_(1, first_samples[i + 1:i + 2].view(1, -1, 1), 1.).view(1, self.classes, 1)

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
                x = np.random.choice(self.output_channels, p=np_prob)
                x = np.array([x])
            else:
                # convert to sample value
                x = torch.max(x, 0)[1][0]
                x = x.cpu()
                x = x.data.numpy()

            o = (x / self.output_channels) * 2. - 1
            generated = np.append(generated, o)

            # set new input
            input = Variable(self.dtype([[o]]), volatile=True)
            # ONE HOT
            # x = Variable(torch.from_numpy(x).type(torch.LongTensor))
            # input.zero_()
            # input = input.scatter_(1, x.view(1, -1, 1), 1.).view(1, self.classes, 1)

            if (i + 1) == 100:
                toc = time.time()
                print("one generating step does take approximately " + str((toc - tic) * 0.01) + " seconds)")

            # progress feedback
            if (i + num_given_samples) % progress_interval == 0:
                if progress_callback is not None:
                    progress_callback(i + num_given_samples, total_samples)

        self.train()
        mu_gen = mu_law_expansion(generated, self.output_channels)
        return mu_gen

    def cpu(self, type=torch.FloatTensor):
        self.context_stack.cpu()
        super().cpu()

parallel_wavenet_default_settings = wavenet_default_settings
parallel_wavenet_default_settings["stacks"] = 3


class ParallelWaveNet(nn.Module):
    def __init__(self, args_dict=parallel_wavenet_default_settings):

        super().__init__()

        self.dtype = args_dict["dtype"]
        self.receptive_field = 0
        self.stacks = nn.ModuleList()

        stack_dict = args_dict.copy()
        stack_dict["output_length"] = 1

        for i in range(args_dict["stacks"]):
            self.stacks.append(WaveNetModel(stack_dict))
            self.receptive_field += self.stacks[i].receptive_field

        self._output_length = 1
        self.output_length = args_dict["output_length"]

    @property
    def output_length(self):
        return self._output_length

    @property
    def input_length(self):
        return self.receptive_field + self._output_length - 1

    @output_length.setter
    def output_length(self, value):
        self._output_length = value
        total_receptive_field = 0
        for i, w in enumerate(self.stacks):
            total_receptive_field += w.receptive_field
            w.output_length = self.input_length - total_receptive_field

    def forward(self, z):
        '''
        :param z: (n, c, l), where n is the minibatch size, c is the number of channels (usually 1) and l is self.input_length
        '''

        mu_tot = torch.zeros_like(z)
        s_tot = torch.zeros_like(z)
        x = z

        for stack in self.stacks:
            input = x[:, :, :-1]
            result = stack.wavenet(input, dilation_func=stack.wavenet_dilate)

            result = result[:, :, -stack.output_length:]
            mu = result[:, 0, :].unsqueeze(1)
            s = result[:, 1, :].unsqueeze(1)

            x = x[:, :, -stack.output_length:]
            mu_tot = mu_tot[:, :, -stack.output_length:]
            s_tot = s_tot[:, :, -stack.output_length:]

            s_exp = torch.exp(s)
            x = x * s_exp + mu
            mu_tot = mu_tot * s_exp + mu
            s_tot += s

        if x.size(2) != self.output_length:
            x = x[:, :, -self.output_length:]
            mu_tot = mu_tot[:, :, -self.output_length:]
            s_tot = s_tot[:, :, -self.output_length:]

        return x, mu_tot, s_tot

    def generate(self,
                 num_samples,
                 progress_callback=None,
                 progress_interval=100):

        self.eval()

        u = self.dtype(1, 1, self.receptive_field + num_samples)
        u.uniform_(1e-5, 1. - 1e-5)
        z = torch.log(u) - torch.log(1. - u)
        z = Variable(z, requires_grad=False)

        generated = None
        generation_steps = math.ceil(num_samples / self.output_length)

        for step in range(generation_steps):
            print("step " + str(step+1) + "/" + str(generation_steps))
            z_position = step * self.output_length
            z_input = z[:, :, z_position:z_position + self.input_length]
            res, mu, s = self(z_input)
            if generated is None:
                generated = res
            else:
                generated = torch.cat([generated, res], dim=2)

        return generated

    def parameter_count(self):
        return sum([stack.parameter_count() for stack in self.stacks])


conditioned_parallel_wavenet_default_settings = wavenet_default_settings
conditioned_parallel_wavenet_default_settings["blocks"] = 1
conditioned_parallel_wavenet_default_settings["conditioning_channels"] = [16, 32, 16]
conditioned_parallel_wavenet_default_settings["file_encoding_channels"] = [32, 16]
conditioned_parallel_wavenet_default_settings["conditioning_period"] = 128


class ParallelWaveNetWithConditioning(ParallelWaveNet):
    def __init__(self, args_dict=conditioned_parallel_wavenet_default_settings):
        nn.Module.__init(self)

        self.dtype = args_dict["dtype"]
        self.receptive_field = 0
        self.conditioning_channels = args_dict["conditioning_channels"]
        self.conditioning_period = args_dict["conditioning_period"]
        self.stacks = nn.ModuleList()

        stack_dict = args_dict.copy
        stack_dict["output_length"] = 1

        for i in range(args_dict['stacks']):
            self.stacks.append(WaveNetModelWithConditioning(stack_dict))
            self.receptive_field += self.stacks[i].receptive_field

        self._output_length = 1
        self.output_length = args_dict['output_length']

    def forward(self, input):
        '''
        :param z: (n, c, l), where n is the minibatch size, c is the number of channels (usually 1) and l is self.input_length
        '''

        z, conditioning, file_encoding, offset = input
        activation_input = {'x': None, 'conditioning': conditioning, 'offset': offset}

        mu_tot = torch.zeros_like(z)
        s_tot = torch.zeros_like(z)
        x = z

        for stack in self.stacks:
            input = x[:, :, :-1]
            result = stack.wavenet(input,
                                   dilation_func=stack.wavenet_dilate,
                                   activation_input=activation_input)

            result = result[:, :, -stack.output_length:]
            mu = result[:, 0, :].unsqueeze(1)
            s = result[:, 1, :].unsqueeze(1)

            x = x[:, :, -stack.output_length:]
            mu_tot = mu_tot[:, :, -stack.output_length:]
            s_tot = s_tot[:, :, -stack.output_length:]

            s_exp = torch.exp(s)
            x = x * s_exp + mu
            mu_tot = mu_tot * s_exp + mu
            s_tot += s

        if x.size(2) != self.output_length:
            x = x[:, :, -self.output_length:]
            mu_tot = mu_tot[:, :, -self.output_length:]
            s_tot = s_tot[:, :, -self.output_length:]

        return x, mu_tot, s_tot

    def generate(self,
                 num_samples,
                 conditioning=None,
                 progress_callback=None,
                 progress_interval=100):

        self.eval()

        u = self.dtype(1, 1, self.receptive_field + num_samples)
        u.uniform_(1e-5, 1. - 1e-5)
        z = torch.log(u) - torch.log(1. - u)
        z = Variable(z, requires_grad=False)

        generated = None
        generation_steps = math.ceil(num_samples / self.output_length)

        for step in range(generation_steps):
            print("step " + str(step+1) + "/" + str(generation_steps))
            z_position = step * self.output_length
            z_input = z[:, :, z_position:z_position + self.input_length]
            res, mu, s = self(z_input)
            if generated is None:
                generated = res
            else:
                generated = torch.cat([generated, res], dim=2)

        return generated


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
