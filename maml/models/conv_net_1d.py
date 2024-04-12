from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from maml.models.model import Model


def weight_init(module):
    if (isinstance(module, torch.nn.Linear)
        or isinstance(module, torch.nn.Conv2d)):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()


class ConvModelOneDimensional(Model):
    """
    NOTE: difference to tf implementation: batch norm scaling is enabled here
    TODO: enable 'non-transductive' setting as per
          https://arxiv.org/abs/1803.02999
    """
    def __init__(self, input_channels, output_size, num_channels=64,
                 kernel_size=5, padding=2, nonlinearity=F.relu,
                 use_max_pool=False, img_side_len=28, verbose=False):
        super(ConvModelOneDimensional, self).__init__()
        self._input_channels = input_channels
        self._output_size = output_size
        self._num_channels = num_channels
        self._kernel_size = kernel_size
        self._nonlinearity = nonlinearity
        self._use_max_pool = use_max_pool
        self._padding = padding
        self._bn_affine = False
        self._reuse = False
        self._verbose = verbose

        assert isinstance(kernel_size, int)


        if self._use_max_pool:
            self._conv_stride = 1
            self._features_size = 1
            self._dialation = 3
            self._pooling_size = 4
            self._padding = self._padding * (self._dialation + 1)
            self.features = torch.nn.Sequential(OrderedDict([
                ('layer1_conv', torch.nn.Conv1d(self._input_channels,
                                                self._num_channels,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding,
                                                dilation=self._dialation)),
                ('layer1_bn', torch.nn.BatchNorm1d(self._num_channels,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer1_max_pool', torch.nn.MaxPool1d(kernel_size=self._pooling_size,
                                                       stride=self._pooling_size)),
                ('layer1_relu', torch.nn.ReLU(inplace=True)),
                ('layer2_conv', torch.nn.Conv1d(self._num_channels,
                                                self._num_channels*2,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding,
                                                dilation=self._dialation)),
                ('layer2_bn', torch.nn.BatchNorm1d(self._num_channels*2,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer2_max_pool', torch.nn.MaxPool1d(kernel_size=self._pooling_size,
                                                       stride=self._pooling_size)),
                ('layer2_relu', torch.nn.ReLU(inplace=True)),
                ('layer3_conv', torch.nn.Conv1d(self._num_channels*2,
                                                self._num_channels*4,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding,
                                                dilation=self._dialation)),
                ('layer3_bn', torch.nn.BatchNorm1d(self._num_channels*4,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer3_max_pool', torch.nn.MaxPool1d(kernel_size=self._pooling_size,
                                                       stride=self._pooling_size)),
                ('layer3_relu', torch.nn.ReLU(inplace=True)),
                ('layer4_conv', torch.nn.Conv1d(self._num_channels*4,
                                                self._num_channels*8,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding,
                                                dilation=self._dialation)),
                ('layer4_bn', torch.nn.BatchNorm1d(self._num_channels*8,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer4_max_pool', torch.nn.MaxPool1d(kernel_size=self._pooling_size,
                                                       stride=self._pooling_size)),
                ('layer4_relu', torch.nn.ReLU(inplace=True)),
            ]))
        else:
            self._conv_stride = 4
            self._features_size = 1
            self._dialation = 3
            self._pooling_size = 4
            self._padding = (self._dialation * (self._kernel_size - 1) - self._conv_stride + 1) // 2
            self._features_size = (img_side_len // 14)**2
            self.features = torch.nn.Sequential(OrderedDict([
                ('layer1_conv', torch.nn.Conv1d(self._input_channels,
                                                self._num_channels,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding)),
                ('layer1_bn', torch.nn.BatchNorm1d(self._num_channels,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer1_relu', torch.nn.ReLU(inplace=True)),
                ('layer2_conv', torch.nn.Conv1d(self._num_channels,
                                                self._num_channels*2,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding)),
                ('layer2_bn', torch.nn.BatchNorm1d(self._num_channels*2,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer2_relu', torch.nn.ReLU(inplace=True)),
                ('layer3_conv', torch.nn.Conv1d(self._num_channels*2,
                                                self._num_channels*4,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding)),
                ('layer3_bn', torch.nn.BatchNorm1d(self._num_channels*4,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer3_relu', torch.nn.ReLU(inplace=True)),
                ('layer4_conv', torch.nn.Conv1d(self._num_channels*4,
                                                self._num_channels*8,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding)),
                ('layer4_bn', torch.nn.BatchNorm1d(self._num_channels*8,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer4_relu', torch.nn.ReLU(inplace=True)),
            ]))

        self.classifier = torch.nn.Sequential(OrderedDict([
            ('fully_connected', torch.nn.Linear(self._num_channels*8,
                                                self._output_size))
        ]))
        self.apply(weight_init)

    def forward(self, task, params=None, embeddings=None):
        if not self._reuse and self._verbose: print('='*10 + ' Model ' + '='*10)
        if params is None:
            params = OrderedDict(self.named_parameters())

        x = task.x
        if not self._reuse and self._verbose: print('input size: {}'.format(x.size()))
        for layer_name, layer in self.features.named_children():
            layer: nn.Conv1d
            weight = params.get('features.' + layer_name + '.weight', None)
            bias = params.get('features.' + layer_name + '.bias', None)
            if 'conv' in layer_name:
                x = F.conv1d(x, 
                             weight=weight,
                             bias=bias,
                             stride=self._conv_stride,
                             padding=self._padding,
                             dilation=layer.dilation)

            elif 'bn' in layer_name:
                x = F.batch_norm(x, 
                                 weight=weight,
                                 bias=bias,
                                 running_mean=layer.running_mean,
                                 running_var=layer.running_var,
                                 training=True)
            elif 'max_pool' in layer_name:
                x = F.max_pool1d(x, kernel_size=3, stride=3)
            elif 'relu' in layer_name:
                x = F.relu(x)
            elif 'fully_connected' in layer_name:
                break
            else:
                raise ValueError('Unrecognized layer {}'.format(layer_name))
            if not self._reuse and self._verbose: print('{}: {}'.format(layer_name, x.size()))

        # in maml network the conv maps are average pooled
        # TODO: something here
        x = x.view(x.size(0), self._num_channels*8, self._features_size)
        if not self._reuse and self._verbose: print('reshape to: {}'.format(x.size()))
        x = torch.mean(x, dim=2)
        if not self._reuse and self._verbose: print('reduce mean: {}'.format(x.size()))
        logits = F.linear(
            x, weight=params['classifier.fully_connected.weight'],
            bias=params['classifier.fully_connected.bias'])
        if not self._reuse and self._verbose: print('logits size: {}'.format(logits.size()))
        if not self._reuse and self._verbose: print('='*27)
        self._reuse = True
        return logits
