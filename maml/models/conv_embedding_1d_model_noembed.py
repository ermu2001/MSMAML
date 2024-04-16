from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvEmbeddingOneDimensionalNoEmbedModel(torch.nn.Module):

    _audio_embed_stride = 320

    _modality2task_names = {
        'audio': ['ESC50', 'CIFAR1001d'],
    }

    def task_name2modelity(self, task_name):
        for modality, task_names in self._modality2task_names.items():
            if task_name in task_names:
                return modality
        raise ValueError(f'not valid task name {task_name}')



    def __init__(self, input_size, output_size, embedding_dims,
                 hidden_size=128, num_layers=1,
                 convolutional=False, num_conv=4, num_channels=32, num_channels_max=256,
                 rnn_aggregation=False, linear_before_rnn=False, 
                 embedding_pooling='max', batch_norm=True, avgpool_after_conv=True,
                 num_sample_embedding=0, sample_embedding_file='embedding.hdf5',
                 img_size=(1, 28, 28), verbose=False):

        super(ConvEmbeddingOneDimensionalNoEmbedModel, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._embedding_dims = embedding_dims
        self._bidirectional = True
        self._device = 'cpu'
        self._convolutional = convolutional
        self._num_conv = num_conv
        self._num_channels = num_channels
        self._num_channels_max = num_channels_max
        self._batch_norm = batch_norm
        self._img_size = img_size
        self._rnn_aggregation = rnn_aggregation
        self._embedding_pooling = embedding_pooling
        self._linear_before_rnn = linear_before_rnn
        self._embeddings_array = []
        self._num_sample_embedding = num_sample_embedding
        self._sample_embedding_file = sample_embedding_file
        self._avgpool_after_conv = avgpool_after_conv
        self._reuse = False
        self._verbose = verbose
        # TODO: plz do refactor this code here...
        # import pdb; pdb.set_trace();
        if self._convolutional:
            self.embeddings = nn.ModuleDict({
                'audio': torch.nn.Conv1d(1, # audio has 1 channel input
                                        self._num_channels,
                                        self._audio_embed_stride,
                                        stride=self._audio_embed_stride,),
            })
            self._conv_stride = 1
            self._features_size = 1
            self._kernel_size = 3
            self._dialation = 1
            # self._padding = "same" # not for strided conv
            self._padding = (self._dialation * (self._kernel_size - 1) - self._conv_stride + 1) // 2 # this ensures the output to be devided by stride
            conv_list = OrderedDict([])
            num_ch = [self._num_channels] + [self._num_channels*2**i for i in range(self._num_conv)]
            num_ch = [min(num_channels_max, ch) for ch in num_ch]
            for i in range(self._num_conv):
                conv_list.update({'conv{}'.format(i+1): torch.nn.Conv1d(num_ch[i],
                                                                        num_ch[i+1],
                                                                        self._kernel_size,
                                                                        stride=self._conv_stride,
                                                                        padding=self._padding,
                                                                        dilation=self._dialation)})
                if self._batch_norm:
                    conv_list.update({'bn{}'.format(i+1): torch.nn.BatchNorm1d(num_ch[i+1], momentum=0.001)})
                conv_list.update({'relu{}'.format(i+1): torch.nn.ReLU(inplace=True)})
            self.conv = torch.nn.Sequential(conv_list)
            self._num_layer_per_conv = len(conv_list) // self._num_conv

            if self._linear_before_rnn:
                linear_input_size = self.compute_input_size(
                    1, 3, 2, self.conv[self._num_layer_per_conv*(self._num_conv-1)].out_channels)
                rnn_input_size = 128
            else:
                if self._avgpool_after_conv:
                    rnn_input_size = self.conv[self._num_layer_per_conv*(self._num_conv-1)].out_channels
                else:
                    rnn_input_size = self.compute_input_size(
                        1, 3, 2, self.conv[self._num_layer_per_conv*(self._num_conv-1)].out_channels)
        else:
            rnn_input_size = int(input_size)

        if self._rnn_aggregation:
            assert False
            if self._linear_before_rnn:
                self.linear = torch.nn.Linear(linear_input_size, rnn_input_size)
                self.relu_after_linear = torch.nn.ReLU(inplace=True)
            self.rnn = torch.nn.GRU(rnn_input_size, hidden_size,
                                    num_layers, bidirectional=self._bidirectional)
            embedding_input_size = hidden_size*(2 if self._bidirectional else 1)
        else:
            self.rnn = None
            embedding_input_size = hidden_size
            self.linear = torch.nn.Linear(rnn_input_size, embedding_input_size)
            self.relu_after_linear = torch.nn.ReLU(inplace=True)

        self._embeddings = torch.nn.ModuleList()
        for dim in embedding_dims:
            self._embeddings.append(torch.nn.Linear(embedding_input_size, dim))

    def compute_input_size(self, p, k, s, ch):
        current_img_size = self._img_size[1]
        for _ in range(self._num_conv):
            current_img_size = (current_img_size+2*p-k)//s+1
        return ch * int(current_img_size) ** 2


    def embed(self, x, task_name, params):
            modality = self.task_name2modelity(task_name)
            layer = self.embeddings[modality]
            weight = params.get('embeddings.' + modality + '.weight', None)
            bias = params.get('features.' + modality + '.bias', None)
            if modality == 'audio':
                layer: nn.Conv1d
                x = F.conv1d(x, 
                             weight=weight,
                             bias=bias,
                             stride=layer.stride,
                             padding=layer.padding,
                             dilation=layer.dilation)
            elif modality == 'image':
                layer: nn.Conv2d
                x = F.conv2d(x, 
                             weight=weight,
                             bias=bias,
                             stride=layer.stride,
                             padding=layer.padding,
                             dilation=layer.dilation)
                x = x.view(x.shape[0], x.shape[1], -1) # keep the bsz, dim, reshape a sequence 
            elif modality == 'text':
                layer: nn.Conv1d
                x = F.conv1d(x, 
                             weight=weight,
                             bias=bias,
                             stride=layer.stride,
                             padding=layer.padding,
                             dilation=layer.dilation)
            else:
                raise ValueError(f'not valid task name {task_name}')
            return x


    def forward(self, task, params=None):
        if not self._reuse and self._verbose: print('='*8 + ' Emb Model ' + '='*8)
        if params is None:
            params = OrderedDict(self.named_parameters())
        # import pdb; pdb.set_trace();
        if self._convolutional:
            x = task.x
            task_name = task.task_info
            if not self._reuse and self._verbose: print('input size: {}'.format(x.size()))
            x = self.embed(x, task_name, params)
            if not self._reuse and self._verbose: print('embed size: {}'.format(x.size()))
            for layer_name, layer in self.conv.named_children():
                layer: nn.Conv1d
                weight = params.get('conv.' + layer_name + '.weight', None)
                bias = params.get('conv.' + layer_name + '.bias', None)
                if 'conv' in layer_name:
                    x = F.conv1d(x,
                                 weight=weight,
                                 bias=bias,
                                 stride=layer.stride,
                                 padding=layer.padding,
                                 dilation=layer.dilation)
                elif 'relu' in layer_name:
                    x = F.relu(x)
                elif 'bn' in layer_name:
                    x = F.batch_norm(x, weight=weight, bias=bias,
                                     running_mean=layer.running_mean,
                                     running_var=layer.running_var,
                                     training=True)
                if not self._reuse and self._verbose: print('{}: {}'.format(layer_name, x.size()))
            if self._avgpool_after_conv:
                x = x.view(x.size(0), x.size(1), -1)
                if not self._reuse and self._verbose: print('reshape to: {}'.format(x.size()))
                x = torch.mean(x, dim=2)
                if not self._reuse and self._verbose: print('reduce mean: {}'.format(x.size()))

            else:
                x = task.x.view(task.x.size(0), -1)
                if not self._reuse and self._verbose: print('flatten: {}'.format(x.size()))
        else:
            x = task.x.view(task.x.size(0), -1)
            if not self._reuse and self._verbose: print('flatten: {}'.format(x.size()))

        if self._rnn_aggregation:
            assert False
            # LSTM input dimensions are seq_len, batch, input_size
            batch_size = 1
            h0 = torch.zeros(self._num_layers*(2 if self._bidirectional else 1),
                             batch_size, self._hidden_size, device=self._device)
            if self._linear_before_rnn: 
                x = F.relu(self.linear(x))
            inputs = x.view(x.size(0), 1, -1)
            output, hn = self.rnn(inputs, h0)
            if self._bidirectional:
                N, B, H = output.shape
                output = output.view(N, B, 2, H // 2)
                embedding_input = torch.cat([output[-1, :, 0], output[0, :, 1]], dim=1)

        else:
            inputs = F.relu(self.linear(x).view(1, x.size(0), -1).transpose(1, 2))
            if not self._reuse and self._verbose: print('fc: {}'.format(inputs.size()))
            if self._embedding_pooling == 'max':
                embedding_input = F.max_pool1d(inputs, x.size(0)).view(1, -1)
            elif self._embedding_pooling == 'avg':
                embedding_input = F.avg_pool1d(inputs, x.size(0)).view(1, -1)
            else:
                raise NotImplementedError
            if not self._reuse and self._verbose: print('reshape after {}pool: {}'.format(
                self._embedding_pooling, embedding_input.size()))

        # randomly sample embedding vectors
        if not self._num_sample_embedding == 0:
            self._embeddings_array.append(embedding_input.cpu().clone().detach().numpy())
            if len(self._embeddings_array) >= self._num_sample_embedding:
                if self._sample_embedding_file.split('.')[-1] == 'hdf5':
                    import h5py
                    f = h5py.File(self._sample_embedding_file, 'w')
                    f['embedding'] = np.squeeze(np.stack(self._embeddings_array))
                    f.close()
                elif self._sample_embedding_file.split('.')[-1] == 'pt':
                    torch.save(np.squeeze(np.stack(self._embeddings_array)),
                               self._sample_embedding_file)
                else:
                    raise NotImplementedError

        out_embeddings = []
        for i, embedding in enumerate(self._embeddings):
            embedding_vec = embedding(embedding_input)
            out_embeddings.append(embedding_vec)
            if not self._reuse and self._verbose: print('emb vec {} size: {}'.format(
                i+1, embedding_vec.size()))
        if not self._reuse and self._verbose: print('='*27)
        self._reuse = True
        return out_embeddings

    def to(self, device, **kwargs):
        self._device = device
        super(ConvEmbeddingOneDimensionalNoEmbedModel, self).to(device, **kwargs)
