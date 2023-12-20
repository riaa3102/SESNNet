"""
File:
    model/ArtificialBlock.py

Description:
    Defines the FCBlock class, Conv2dBlock class, and the ResConv2dBlock class.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from pathlib import Path
from typing import Optional, List, Union, Tuple
from src.model.SpikingLayer import Linear1d, ScaleLayer, Upsampling2d
from src.model.constants import ActivationParameters


class FCBlock(nn.Module):
    """Class that implements the ANN layer.
    """

    def __init__(self, input_dim: int, output_dim: int, activation_fn: Optional[str],
                 weight_init: dict, scale_flag: bool, scale_factor: float, bn_flag: bool,
                 dropout_flag: bool, dropout_p: float, device, dtype: torch.dtype, layer_index: int) -> None:

        super(FCBlock, self).__init__()

        self.device = device
        self.dtype = dtype

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_index = layer_index

        # Init block
        # ------------
        self.fc_block = nn.Sequential()

        self.fc_block.add_module('linear1d_layer', Linear1d(input_dim=self.input_dim, output_dim=self.output_dim))

        if scale_flag or (not scale_flag and scale_factor != 1):
            self.fc_block.add_module('scale_layer', ScaleLayer(scale_factor=scale_factor, scale_flag=scale_flag))

        if bn_flag:
            self.fc_block.add_module('bn_layer', nn.BatchNorm1d(self.output_dim,
                                                                # eps=1e-5,
                                                                # momentum=0.1,
                                                                # affine=True,
                                                                # track_running_stats=True
                                                                ))

        if dropout_flag:
            self.fc_block.add_module('dropout_layer', nn.Dropout(p=dropout_p))

        if activation_fn == 'sigmoid':
            self.fc_block.add_module('activation_layer', nn.Sigmoid())
        elif activation_fn == 'relu':
            self.fc_block.add_module('activation_layer', nn.ReLU())
        elif activation_fn == 'lrelu':
            self.fc_block.add_module('activation_layer', nn.LeakyReLU(negative_slope=
                                                                      ActivationParameters.negative_slope))
        elif activation_fn == 'prelu':
            self.fc_block.add_module('activation_layer', nn.PReLU())
        elif activation_fn == 'tanh':
            self.fc_block.add_module('activation_layer', nn.Tanh())

        # Reset Parameters
        self.init_weights(weight_init)

    def forward(self, x):
        """Method that defines the performed computation during the forward pass.
        """

        return self.fc_block(x)

    def init_weights(self, weight_init: dict):
        """Method that handles the linear parameters' initialization.

        Parameters
        ----------
        weight_init: dict
            Initialization specifications' dictionary.
        """
        if weight_init['weight_init_dist'] == 'normal_':
            w_std_ = 1. / np.sqrt(self.input_dim)
            nn.init.normal_(self.fc_block.linear1d_layer.weight.data,
                            mean=weight_init['weight_mean'], std=w_std_ * weight_init['weight_std'])

        elif weight_init['weight_init_dist'] == 'uniform_':
            w_bound_ = 1. / np.sqrt(self.input_dim)
            nn.init.uniform_(self.fc_block.linear1d_layer.weight.data,
                             a=-w_bound_ * weight_init['weight_gain'], b=w_bound_ * weight_init['weight_gain'])

        elif weight_init['weight_init_dist'] == 'kaiming_normal_':
            nn.init.kaiming_normal_(self.fc_block.linear1d_layer.weight.data, a=weight_init['weight_gain'], mode='fan_in')

        elif weight_init['weight_init_dist'] == 'kaiming_uniform_':
            nn.init.kaiming_uniform_(self.fc_block.linear1d_layer.weight.data, a=weight_init['weight_gain'], mode='fan_in')

        elif weight_init['weight_init_dist'] == 'xavier_uniform_':
            nn.init.xavier_uniform_(self.fc_block.linear1d_layer.weight.data, gain=weight_init['weight_gain'])


class Conv2dBlock(nn.Module):
    """Class that implements the SNN spiking layer using LIF neuron model.
    """

    def __init__(self, input_dim: int, output_dim: int, input_channels: int, output_channels: int,
                 kernel_size: Tuple[int, int], stride: Tuple[int, int], padding: Tuple[int, int],
                 dilation: Tuple[int, int], bias: bool, nb_steps: int, activation_fn: Optional[str],
                 weight_init: dict, upsample_flag: bool, upsample_mode: str, scale_flag: bool, scale_factor: float,
                 bn_flag: bool, dropout_flag: bool, dropout_p: float, device, dtype: torch.dtype,
                 layer_index: int) -> None:

        super(Conv2dBlock, self).__init__()

        self.device = device
        self.dtype = dtype

        # if this is the input layer : input_channels = 1
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_index = layer_index
        self.upsample_flag = upsample_flag
        self.upsample_mode = upsample_mode

        # Init block
        # ------------
        self.conv2d_block = nn.Sequential()

        if upsample_flag:
            self.conv2d_block.add_module('upsample2d_layer', Upsampling2d(input_dim, nb_steps, upsample_mode))

        self.conv2d_block.add_module('conv2d_layer',
                                     nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                                               kernel_size=kernel_size, stride=stride, padding=padding,
                                               padding_mode='replicate', dilation=dilation, bias=self.bias))

        if scale_flag or (not scale_flag and scale_factor != 1):
            self.conv2d_block.add_module('scale_layer', ScaleLayer(scale_factor=scale_factor, scale_flag=scale_flag))

        if bn_flag:
            self.conv2d_block.add_module('bn_layer', nn.BatchNorm2d(self.output_channels))

        if dropout_flag:
            self.conv2d_block.add_module('dropout_layer', nn.Dropout(p=dropout_p))

        if activation_fn == 'sigmoid':
            self.conv2d_block.add_module('activation_layer', nn.Sigmoid())
        elif activation_fn == 'relu':
            self.conv2d_block.add_module('activation_layer', nn.ReLU())
        elif activation_fn == 'lrelu':
            self.conv2d_block.add_module('activation_layer', nn.LeakyReLU(negative_slope=
                                                                          ActivationParameters.negative_slope))
        elif activation_fn == 'prelu':
            self.conv2d_block.add_module('activation_layer', nn.PReLU())
        elif activation_fn == 'tanh':
            self.conv2d_block.add_module('activation_layer', nn.Tanh())

        # Reset Parameters
        self.init_weights(weight_init)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method that defines the performed computation during the forward pass.
        """

        return self.conv2d_block(x)

    def init_weights(self, weight_init: dict):
        """Method that handles the convolution parameters' initialization.

        Parameters
        ----------
        weight_init: dict
            Initialization specifications' dictionary.
        """
        if weight_init['weight_init_dist'] == 'normal_':
            w_std_ = weight_init['weight_std'] / np.sqrt(self.input_channels * np.prod(self.kernel_size))
            nn.init.normal_(self.conv2d_block.conv2d_layer.weight.data, mean=weight_init['weight_mean'], std=w_std_)

        elif weight_init['weight_init_dist'] == 'uniform_':
            w_bound_ = weight_init['weight_gain'] / np.sqrt(self.input_channels * np.prod(self.kernel_size))
            nn.init.uniform_(self.conv2d_block.conv2d_layer.weight.data, a=-w_bound_, b=w_bound_)

        elif weight_init['weight_init_dist'] == 'kaiming_normal_':
            nn.init.kaiming_normal_(self.conv2d_block.conv2d_layer.weight.data, a=weight_init['weight_gain'], mode='fan_in')

        elif weight_init['weight_init_dist'] == 'kaiming_uniform_':
            nn.init.kaiming_uniform_(self.conv2d_block.conv2d_layer.weight.data, a=weight_init['weight_gain'], mode='fan_in')

        elif weight_init['weight_init_dist'] == 'xavier_uniform_':
            nn.init.xavier_uniform_(self.conv2d_block.conv2d_layer.weight.data, gain=weight_init['weight_gain'])


class ResConv2dBlock(nn.Module):
    """Class that implements the ANN layer.
    """

    def __init__(self, input_dim: int, output_dim: int, input_channels: int, output_channels: int,
                 kernel_size: Tuple[int, int], stride: Tuple[int, int], padding: Tuple[int, int],
                 dilation: Tuple[int, int], bias: bool, activation_fn: Optional[str], weight_init: dict,
                 scale_flag: bool, scale_factor: float, bn_flag: bool, dropout_flag: bool, dropout_p: float,
                 residual_skip_connection_type: str,
                 device, dtype: torch.dtype, layer_index: int) -> None:

        super(ResConv2dBlock, self).__init__()

        self.device = device
        self.dtype = dtype

        # if this is the input layer : input_channels = 1
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual_skip_connection_type = residual_skip_connection_type
        self.layer_index = layer_index

        # Init block_1
        # ------------
        self.conv2d_block_1 = nn.Sequential()

        self.conv2d_block_1.add_module('conv2d_layer',
                                       nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                                 padding_mode='replicate', dilation=dilation, bias=self.bias))

        if scale_flag or (not scale_flag and scale_factor != 1):
            self.conv2d_block_1.add_module('scale_layer', ScaleLayer(scale_factor=scale_factor, scale_flag=scale_flag))

        if bn_flag:
            self.conv2d_block_1.add_module('bn_layer', nn.BatchNorm2d(self.output_channels))

        if dropout_flag:
            self.conv2d_block_1.add_module('dropout_layer', nn.Dropout(p=dropout_p))

        if activation_fn == 'sigmoid':
            self.conv2d_block_1.add_module('activation_layer', nn.Sigmoid())
        elif activation_fn == 'relu':
            self.conv2d_block_1.add_module('activation_layer', nn.ReLU())
        elif activation_fn == 'lrelu':
            self.conv2d_block_1.add_module('activation_layer', nn.LeakyReLU(negative_slope=
                                                                            ActivationParameters.negative_slope))
        elif activation_fn == 'prelu':
            self.conv2d_block_1.add_module('activation_layer', nn.PReLU())
        elif activation_fn == 'tanh':
            self.conv2d_block_1.add_module('activation_layer', nn.Tanh())

        # Init block_2
        # ------------
        self.conv2d_block_2 = nn.Sequential()

        self.conv2d_block_2.add_module('conv2d_layer',
                                       nn.Conv2d(in_channels=output_channels, out_channels=output_channels,
                                                 kernel_size=kernel_size, stride=1, padding=padding,
                                                 padding_mode='replicate', dilation=dilation, bias=self.bias))

        if scale_flag or (not scale_flag and scale_factor != 1):
            self.conv2d_block_2.add_module('scale_layer', ScaleLayer(scale_factor=scale_factor, scale_flag=scale_flag))

        if bn_flag:
            self.conv2d_block_2.add_module('bn_layer', nn.BatchNorm2d(self.output_channels))

        if dropout_flag:
            self.conv2d_block_2.add_module('dropout_layer', nn.Dropout(p=dropout_p))

        if activation_fn == 'sigmoid':
            self.conv2d_block_2.add_module('activation_layer', nn.Sigmoid())
        elif activation_fn == 'relu':
            self.conv2d_block_2.add_module('activation_layer', nn.ReLU())
        elif activation_fn == 'lrelu':
            self.conv2d_block_2.add_module('activation_layer', nn.LeakyReLU(negative_slope=
                                                                            ActivationParameters.negative_slope))
        elif activation_fn == 'prelu':
            self.conv2d_block_2.add_module('activation_layer', nn.PReLU())
        elif activation_fn == 'tanh':
            self.conv2d_block_2.add_module('activation_layer', nn.Tanh())

        # Init block_skp
        # --------------
        self.init_block_skp = False
        self.conv2d_block_skp = nn.Identity()

        if stride[0] != 1 or stride[1] != 1 or input_channels != output_channels:
            self.init_block_skp = True
            self.conv2d_block_skp = nn.Sequential()
            self.conv2d_block_skp.add_module('conv2d_layer',
                                             nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                                                       kernel_size=kernel_size, stride=stride, padding=padding,
                                                       padding_mode='replicate', dilation=dilation, bias=self.bias))

            if scale_flag or (not scale_flag and scale_factor != 1):
                self.conv2d_block_skp.add_module('scale_layer', ScaleLayer(scale_factor=scale_factor,
                                                                           scale_flag=scale_flag))

            if bn_flag:
                self.conv2d_block_skp.add_module('bn_layer', nn.BatchNorm2d(self.output_channels))

            if dropout_flag:
                self.conv2d_block_skp.add_module('dropout_layer', nn.Dropout(p=dropout_p))

            if activation_fn == 'sigmoid':
                self.conv2d_block_skp.add_module('activation_layer', nn.Sigmoid())
            elif activation_fn == 'relu':
                self.conv2d_block_skp.add_module('activation_layer', nn.ReLU())
            elif activation_fn == 'lrelu':
                self.conv2d_block_skp.add_module('activation_layer', nn.LeakyReLU(negative_slope=
                                                                                  ActivationParameters.negative_slope))
            elif activation_fn == 'prelu':
                self.conv2d_block_skp.add_module('activation_layer', nn.PReLU())
            elif activation_fn == 'tanh':
                self.conv2d_block_skp.add_module('activation_layer', nn.Tanh())

        # Reset Parameters
        self.init_weights(weight_init)

    def forward(self, x: torch.Tensor):
        """Method that defines the performed computation during the forward pass.
        """

        identity = x

        output_records_1 = self.conv2d_block_1(x)

        output_records_2 = self.conv2d_block_2(output_records_1)

        if self.init_block_skp:
            identity, membrane_potential_records_skp = self.conv2d_block_skp(identity)

        return self.res_skip_connection(output_records_2, identity)

    def res_skip_connection(self, output_records, identity):
        if self.residual_skip_connection_type == 'add_':
            output_records = identity + output_records
        elif self.residual_skip_connection_type == 'and_':
            output_records = identity * output_records
        elif self.residual_skip_connection_type == 'iand_':
            output_records = identity * (1. - output_records)

        return output_records

    def init_weights(self, weight_init: dict):
        """Method that handles the convolution parameters' initialization.

        Parameters
        ----------
        weight_init: dict
            Initialization specifications' dictionary.
        """
        if weight_init['weight_init_dist'] == 'normal_':
            w_std_ = weight_init['weight_std'] / np.sqrt(self.input_channels * np.prod(self.kernel_size))
            nn.init.normal_(self.conv2d_block_1.conv2d_layer.weight.data, mean=weight_init['weight_mean'], std=w_std_)
            nn.init.normal_(self.conv2d_block_2.conv2d_layer.weight.data, mean=weight_init['weight_mean'], std=w_std_)
            if self.init_block_skp:
                nn.init.normal_(self.conv2d_block_skp.conv2d_layer.weight.data,
                                mean=weight_init['weight_mean'], std=w_std_)

        elif weight_init['weight_init_dist'] == 'uniform_':
            w_bound_ = ['weight_gain'] / np.sqrt(self.input_channels * np.prod(self.kernel_size))
            nn.init.uniform_(self.conv2d_block_1.conv2d_layer.weight.data, a=-w_bound_, b=w_bound_)
            nn.init.uniform_(self.conv2d_block_2.conv2d_layer.weight.data, a=-w_bound_, b=w_bound_)
            if self.init_block_skp:
                nn.init.uniform_(self.conv2d_block_skp.conv2d_layer.weight.data, a=-w_bound_, b=w_bound_)

        elif weight_init['weight_init_dist'] == 'kaiming_normal_':
            nn.init.kaiming_normal_(self.conv2d_block_1.conv2d_layer.weight.data, a=weight_init['weight_gain'], mode='fan_in')
            nn.init.kaiming_normal_(self.conv2d_block_2.conv2d_layer.weight.data, a=weight_init['weight_gain'], mode='fan_in')
            if self.init_block_skp:
                nn.init.kaiming_normal_(self.conv2d_block_skp.conv2d_layer.weight.data, a=weight_init['weight_gain'], mode='fan_in')

        elif weight_init['weight_init_dist'] == 'kaiming_uniform_':
            nn.init.kaiming_uniform_(self.conv2d_block_1.conv2d_layer.weight.data, a=weight_init['weight_gain'], mode='fan_in')
            nn.init.kaiming_uniform_(self.conv2d_block_2.conv2d_layer.weight.data, a=weight_init['weight_gain'], mode='fan_in')
            if self.init_block_skp:
                nn.init.kaiming_uniform_(self.conv2d_block_skp.conv2d_layer.weight.data,
                                         a=weight_init['weight_gain'], mode='fan_in')

        elif weight_init['weight_init_dist'] == 'xavier_uniform_':
            nn.init.xavier_uniform_(self.conv2d_block_1.conv2d_layer.weight.data, gain=weight_init['weight_gain'])
            nn.init.xavier_uniform_(self.conv2d_block_2.conv2d_layer.weight.data, gain=weight_init['weight_gain'])
            if self.init_block_skp:
                nn.init.xavier_uniform_(self.conv2d_block_skp.conv2d_layer.weight.data, gain=weight_init['weight_gain'])

