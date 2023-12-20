"""
File:
    model/SpikingBlock.py

Description:
    Defines the SpikingFCBlock class, SpikingConv2dBlock class, ReadoutBlock, ReadoutConv2dBlock and the SpikingNetwork class.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Union, Tuple
from src.model.SpikingLayer import Linear1d, ScaleLayer, LIF1d, LI1d, Upsampling2d, LIF2d, LI2d, PLIF2d, PLI2d, \
    KLIF2d, KLI2d, IF2d, I2d
from src.model.SurrogateGradient import SuperSpike, SigmoidDerivative, PiecewiseLinear, ATan


class SpikingFCBlock(nn.Module):
    """Class that implements the SNN spiking linear block.
    """

    def __init__(self, input_dim: int, output_dim: int, nb_steps: int, truncated_bptt_ratio: int,
                 spike_fn: Union[SuperSpike, SigmoidDerivative, PiecewiseLinear, ATan], neuron_model: str,
                 neuron_parameters: dict, weight_init: dict, scale_flag: bool, scale_factor: float, bn_flag: bool,
                 dropout_flag: bool, dropout_p: float, device, dtype: torch.dtype, layer_index: int) -> None:

        super(SpikingFCBlock, self).__init__()

        self.device = device
        self.dtype = dtype

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nb_steps = nb_steps
        self.truncated_bptt_ratio = truncated_bptt_ratio
        self.spike_fn = spike_fn.apply
        self.layer_index = layer_index

        # Init block
        # ------------
        self.fc_block = nn.Sequential()

        self.fc_block.add_module('linear1d_layer', Linear1d(input_dim=self.input_dim, output_dim=self.output_dim))

        if scale_flag or (not scale_flag and scale_factor != 1):
            self.fc_block.add_module('scale_layer', ScaleLayer(scale_factor=scale_factor, scale_flag=scale_flag))

        if bn_flag:
            self.fc_block.add_module('bn_layer', nn.BatchNorm1d(self.output_dim,
                                                                # eps=1e-5, momentum=0.1, affine=True,
                                                                # track_running_stats=True,
                                                                ))
            # nn.init.constant_(self.fc_block.bn_layer.weight.data, val=0.1)
            self.fc_block.bn_layer.bias = None

        if dropout_flag:
            self.fc_block.add_module('dropout_layer', nn.Dropout(p=dropout_p))

        if neuron_model == 'lif':
            self.fc_block.add_module('neuron1d_layer',
                                     LIF1d(output_dim, truncated_bptt_ratio, neuron_parameters['membrane_threshold'],
                                           neuron_parameters['alpha'], neuron_parameters['beta'],
                                           neuron_parameters['train_neuron_parameters'], self.spike_fn,
                                           neuron_parameters['reset_mode'], neuron_parameters['detach_reset'],
                                           neuron_parameters['recurrent_flag'], neuron_parameters['decay_input'],
                                           self.device, self.dtype)
                                     )
        elif neuron_model == 'plif':
            raise NotImplementedError
        elif neuron_model == 'klif':
            raise NotImplementedError
        elif neuron_model == 'if':
            raise NotImplementedError

        # Reset Parameters
        self.init_weights(weight_init)
        if neuron_parameters['recurrent_flag']:
            self.init_recurrent_weights(weight_init)
        if neuron_parameters['train_neuron_parameters']:
            self.init_neuron_parameters(neuron_parameters)

    def forward(self, x: torch.Tensor):
        """Method that defines the performed computation during the forward pass.

        Note
        ------
        Note that this method override the super class method.

        Parameters
        ----------
        x: torch.Tensor
            A tensor of input data.
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
            nn.init.kaiming_normal_(self.fc_block.linear1d_layer.weight.data,
                                    a=weight_init['weight_gain'], mode='fan_in')

        elif weight_init['weight_init_dist'] == 'kaiming_uniform_':
            nn.init.kaiming_uniform_(self.fc_block.linear1d_layer.weight.data,
                                     a=weight_init['weight_gain'], mode='fan_in')

        elif weight_init['weight_init_dist'] == 'xavier_uniform_':
            nn.init.xavier_uniform_(self.fc_block.linear1d_layer.weight.data,
                                    gain=weight_init['weight_gain'])

    def init_recurrent_weights(self, weight_init: dict):
        """Method that handles the recurrent parameters' initialization.

        Parameters
        ----------
        weight_init: dict
            Initialization specifications' dictionary.
        """
        if weight_init['weight_init_dist'] == 'normal_':
            v_std_ = 1. / np.sqrt(self.output_dim)
            nn.init.normal_(self.fc_block.neuron1d_layer.recurrent1d_layer.v.data,
                            mean=weight_init['weight_mean'], std=v_std_ * weight_init['weight_std'])

        elif weight_init['weight_init_dist'] == 'uniform_':
            v_bound_ = 1. / np.sqrt(self.output_dim)
            nn.init.uniform_(self.fc_block.neuron1d_layer.recurrent1d_layer.v.data,
                             a=-v_bound_ * weight_init['weight_gain'], b=v_bound_ * weight_init['weight_gain'])

        elif weight_init['weight_init_dist'] == 'kaiming_normal_':
            nn.init.kaiming_normal_(self.fc_block.neuron1d_layer.recurrent1d_layer.v.data,
                                    a=weight_init['weight_gain'], mode='fan_out')

        elif weight_init['weight_init_dist'] == 'kaiming_uniform_':
            nn.init.kaiming_uniform_(self.fc_block.neuron1d_layer.recurrent1d_layer.v.data,
                                     a=weight_init['weight_gain'], mode='fan_out')

        elif weight_init['weight_init_dist'] == 'xavier_uniform_':
            nn.init.xavier_uniform_(self.fc_block.neuron1d_layer.recurrent1d_layer.v.data,
                                    gain=weight_init['weight_gain'])

    def init_neuron_parameters(self, neuron_parameters: dict):
        """Method that handles the spiking neurons parameters' initialization.

        Parameters
        ----------
        neuron_parameters: dict
            Initialization specifications' dictionary.
        """
        self.fc_block.neuron1d_layer.init_neuron_parameters(neuron_parameters)
        self.fc_block.neuron1d_layer.clamp_neuron_parameters()


class ReadoutFCBlock(nn.Module):
    """Class that implements the SNN readout linear block.
    """

    def __init__(self, input_dim: int, output_dim: int, nb_steps: int, truncated_bptt_ratio: int,
                 spike_fn: Union[SuperSpike, SigmoidDerivative, PiecewiseLinear, ATan], neuron_model: str,
                 neuron_parameters: dict, weight_init: dict, scale_flag: bool, scale_factor: float, device,
                 dtype: torch.dtype, layer_index: int) -> None:

        super(ReadoutFCBlock, self).__init__()

        self.device = device
        self.dtype = dtype

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nb_steps = nb_steps
        self.truncated_bptt_ratio = truncated_bptt_ratio
        self.spike_fn = spike_fn.apply
        self.layer_index = layer_index

        # Init block
        # ------------
        self.fc_block = nn.Sequential()

        self.fc_block.add_module('linear1d_layer', Linear1d(input_dim=self.input_dim, output_dim=self.output_dim))

        if scale_flag or (not scale_flag and scale_factor != 1):
            self.fc_block.add_module('scale_layer', ScaleLayer(scale_factor=scale_factor, scale_flag=scale_flag))

        if neuron_model == 'lif':
            self.fc_block.add_module('mp_neuron1d_layer',
                                     LI1d(output_dim, truncated_bptt_ratio, neuron_parameters['alpha'],
                                          neuron_parameters['beta_out'],
                                          neuron_parameters['train_neuron_parameters'],
                                          neuron_parameters['decay_input'], self.device, self.dtype)
                                     )
        elif neuron_model == 'plif':
            raise NotImplementedError
        elif neuron_model == 'klif':
            raise NotImplementedError
        elif neuron_model == 'if':
            raise NotImplementedError

        # Reset Parameters
        self.init_weights(weight_init)
        if neuron_parameters['train_neuron_parameters']:
            self.init_neuron_parameters(neuron_parameters)

    def forward(self, x: torch.Tensor):
        """Method that defines the performed computation during the forward pass.

        Note
        ------
        Note that this method override the super class method.

        Parameters
        ----------
        x: torch.Tensor
            A tensor of input data.
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
            nn.init.kaiming_normal_(self.fc_block.linear1d_layer.weight.data,
                                    a=weight_init['weight_gain'], mode='fan_in')

        elif weight_init['weight_init_dist'] == 'kaiming_uniform_':
            nn.init.kaiming_uniform_(self.fc_block.linear1d_layer.weight.data,
                                     a=weight_init['weight_gain'], mode='fan_in')

        elif weight_init['weight_init_dist'] == 'xavier_uniform_':
            nn.init.xavier_uniform_(self.fc_block.linear1d_layer.weight.data,
                                    gain=weight_init['weight_gain'])

    def init_neuron_parameters(self, neuron_parameters: dict):
        """Method that handles the readout neurons parameters' initialization.

        Parameters
        ----------
        neuron_parameters: dict
            Initialization specifications' dictionary.
        """
        self.fc_block.mp_neuron1d_layer.init_neuron_parameters(neuron_parameters)
        self.fc_block.mp_neuron1d_layer.clamp_neuron_parameters()


class SpikingConv2dBlock(nn.Module):
    """Class that implements the SNN spiking convolutional block.
    """

    def __init__(self, input_dim: int, output_dim: int, input_channels: int, output_channels: int,
                 kernel_size: Tuple[int, int], stride: Tuple[int, int], padding: Tuple[int, int],
                 dilation: Tuple[int, int], bias: bool, padding_mode: str, pooling_flag: bool, pooling_type: str,
                 pooling_stride: Optional[tuple], pooling_padding: Optional[tuple], nb_steps: int,
                 truncated_bptt_ratio: int, spike_fn: Union[SuperSpike, SigmoidDerivative, PiecewiseLinear, ATan],
                 neuron_model: str, neuron_parameters: dict, weight_init: dict, upsample_flag: bool, upsample_mode: str,
                 scale_flag: bool, scale_factor: float, bn_flag: bool, dropout_flag: bool, dropout_p: float, device,
                 dtype: torch.dtype, layer_index: int) -> None:

        super(SpikingConv2dBlock, self).__init__()

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
        self.padding_mode = padding_mode

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nb_steps = nb_steps
        self.truncated_bptt_ratio = truncated_bptt_ratio
        self.spike_fn = spike_fn.apply
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
                                               dilation=dilation, bias=self.bias, padding_mode=padding_mode))

        if pooling_flag:
            if pooling_type == 'max':
                self.conv2d_block.add_module('pool2d_layer',
                                             nn.MaxPool2d(kernel_size=kernel_size, stride=pooling_stride,
                                                          padding=pooling_padding))
            elif pooling_type == 'avg':
                self.conv2d_block.add_module('pool2d_layer',
                                             nn.AvgPool2d(kernel_size=kernel_size, stride=pooling_stride,
                                                          padding=pooling_padding))

        if scale_flag or (not scale_flag and scale_factor != 1):
            self.conv2d_block.add_module('scale_layer', ScaleLayer(scale_factor=scale_factor, scale_flag=scale_flag))

        if bn_flag:
            self.conv2d_block.add_module('bn_layer', nn.BatchNorm2d(self.output_channels))
            # nn.init.constant_(self.conv2d_block.bn_layer.weight.data, val=0.1)
            self.conv2d_block.bn_layer.bias = None

        if dropout_flag:
            self.conv2d_block.add_module('dropout_layer', nn.Dropout(p=dropout_p))

        if neuron_model == 'lif':
            self.conv2d_block.add_module('neuron2d_layer',
                                         LIF2d(output_dim, output_channels, truncated_bptt_ratio,
                                               neuron_parameters['membrane_threshold'],
                                               neuron_parameters['alpha'], neuron_parameters['beta'],
                                               neuron_parameters['train_neuron_parameters'], self.spike_fn,
                                               neuron_parameters['reset_mode'], neuron_parameters['detach_reset'],
                                               neuron_parameters['recurrent_flag'], neuron_parameters['decay_input'],
                                               self.device, self.dtype)
                                         )
        elif neuron_model == 'plif':
            self.conv2d_block.add_module('neuron2d_layer',
                                         PLIF2d(output_dim, output_channels, truncated_bptt_ratio,
                                                neuron_parameters['membrane_threshold'],
                                                neuron_parameters['alpha'], neuron_parameters['beta'],
                                                neuron_parameters['train_neuron_parameters'], self.spike_fn,
                                                neuron_parameters['reset_mode'], neuron_parameters['detach_reset'],
                                                neuron_parameters['recurrent_flag'], neuron_parameters['decay_input'],
                                                self.device, self.dtype)
                                         )
        elif neuron_model == 'klif':
            self.conv2d_block.add_module('neuron2d_layer',
                                         KLIF2d(output_dim, output_channels, truncated_bptt_ratio,
                                                neuron_parameters['membrane_threshold'],
                                                neuron_parameters['alpha'], neuron_parameters['beta'], neuron_parameters['k'],
                                                neuron_parameters['train_neuron_parameters'], self.spike_fn,
                                                neuron_parameters['reset_mode'], neuron_parameters['detach_reset'],
                                                neuron_parameters['recurrent_flag'], neuron_parameters['decay_input'],
                                                self.device, self.dtype)
                                         )
        elif neuron_model == 'if':
            self.conv2d_block.add_module('neuron2d_layer',
                                         IF2d(output_dim, output_channels, truncated_bptt_ratio,
                                              neuron_parameters['membrane_threshold'],
                                              neuron_parameters['train_neuron_parameters'], self.spike_fn,
                                              neuron_parameters['reset_mode'], neuron_parameters['detach_reset'],
                                              neuron_parameters['recurrent_flag'], neuron_parameters['decay_input'],
                                              self.device, self.dtype)
                                         )

        # Reset Parameters
        self.init_weights(weight_init)
        if neuron_parameters['recurrent_flag']:
            self.init_recurrent_weights(weight_init)
        if neuron_parameters['train_neuron_parameters']:
            self.init_neuron_parameters(neuron_parameters)

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
            nn.init.kaiming_normal_(self.conv2d_block.conv2d_layer.weight.data,
                                    a=weight_init['weight_gain'], mode='fan_in')

        elif weight_init['weight_init_dist'] == 'kaiming_uniform_':
            nn.init.kaiming_uniform_(self.conv2d_block.conv2d_layer.weight.data,
                                     a=weight_init['weight_gain'], mode='fan_in')

        elif weight_init['weight_init_dist'] == 'xavier_uniform_':
            nn.init.xavier_uniform_(self.conv2d_block.conv2d_layer.weight.data, gain=weight_init['weight_gain'])

    def init_recurrent_weights(self, weight_init: dict):
        """Method that handles the recurrent parameters' initialization.

        Parameters
        ----------
        weight_init: dict
            Initialization specifications' dictionary.
        """
        if weight_init['weight_init_dist'] == 'normal_':
            v_std_ = weight_init['weight_std'] / math.sqrt(self.output_channels)
            nn.init.normal_(self.conv2d_block.neuron2d_layer.recurrent2d_layer.v.data,
                            mean=weight_init['weight_mean'], std=v_std_)

        elif weight_init['weight_init_dist'] == 'uniform_':
            v_bound_ = weight_init['weight_gain'] / math.sqrt(self.output_channels)
            nn.init.uniform_(self.conv2d_block.neuron2d_layer.recurrent2d_layer.v.data, a=-v_bound_, b=v_bound_)

        elif weight_init['weight_init_dist'] == 'kaiming_normal_':
            nn.init.kaiming_normal_(self.conv2d_block.neuron2d_layer.recurrent2d_layer.v.data,
                                    a=weight_init['weight_gain'], mode='fan_out')

        elif weight_init['weight_init_dist'] == 'kaiming_uniform_':
            nn.init.kaiming_uniform_(self.conv2d_block.neuron2d_layer.recurrent2d_layer.v.data,
                                     a=weight_init['weight_gain'], mode='fan_out')

        elif weight_init['weight_init_dist'] == 'xavier_uniform_':
            nn.init.xavier_uniform_(self.conv2d_block.neuron2d_layer.recurrent2d_layer.v.data,
                                    gain=weight_init['weight_gain'])

    def init_neuron_parameters(self, neuron_parameters: dict):
        """Method that handles the spiking neurons parameters' initialization.

        Parameters
        ----------
        neuron_parameters: dict
            Initialization specifications' dictionary.
        """
        self.conv2d_block.neuron2d_layer.init_neuron_parameters(neuron_parameters)
        self.conv2d_block.neuron2d_layer.clamp_neuron_parameters()


class ReadoutConv2dBlock(nn.Module):
    """Class that implements the SNN readout convolutional block.
    """

    def __init__(self, input_dim: int, output_dim: int, input_channels: int, output_channels: int,
                 kernel_size: Tuple[int, int], stride: Tuple[int, int], padding: Tuple[int, int],
                 dilation: Tuple[int, int], bias: bool, padding_mode: str, nb_steps: int, truncated_bptt_ratio: int,
                 neuron_model: str, neuron_parameters: dict, weight_init: dict, upsample_flag: bool, upsample_mode: str,
                 scale_flag: bool, scale_factor: float, device, dtype: torch.dtype, layer_index: int) -> None:

        super(ReadoutConv2dBlock, self).__init__()

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
        self.padding_mode = padding_mode

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nb_steps = nb_steps
        self.truncated_bptt_ratio = truncated_bptt_ratio
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
                                               dilation=dilation, bias=self.bias, padding_mode=padding_mode))

        if scale_flag or (not scale_flag and scale_factor != 1):
            self.conv2d_block.add_module('scale_layer', ScaleLayer(scale_factor=scale_factor, scale_flag=scale_flag))

        if neuron_model == 'lif':
            self.conv2d_block.add_module('mp_neuron2d_layer',
                                         LI2d(output_dim, output_channels, truncated_bptt_ratio,
                                              neuron_parameters['alpha_out'], neuron_parameters['beta_out'],
                                              neuron_parameters['train_neuron_parameters'],
                                              neuron_parameters['decay_input'], self.device, self.dtype)
                                         )
        elif neuron_model == 'plif':
            self.conv2d_block.add_module('mp_neuron2d_layer',
                                         PLI2d(output_dim, output_channels, truncated_bptt_ratio,
                                               neuron_parameters['alpha_out'], neuron_parameters['beta_out'],
                                               neuron_parameters['train_neuron_parameters'],
                                               neuron_parameters['decay_input'], self.device, self.dtype)
                                         )
        elif neuron_model == 'klif':
            self.conv2d_block.add_module('mp_neuron2d_layer',
                                         KLI2d(output_dim, output_channels, truncated_bptt_ratio,
                                               neuron_parameters['alpha_out'], neuron_parameters['beta_out'] ,neuron_parameters['k'],
                                               neuron_parameters['train_neuron_parameters'],
                                               neuron_parameters['decay_input'], self.device, self.dtype)
                                         )
        elif neuron_model == 'if':
            self.conv2d_block.add_module('mp_neuron2d_layer',
                                         I2d(output_dim, output_channels, truncated_bptt_ratio,
                                             neuron_parameters['decay_input'], self.device, self.dtype)
                                         )

        # Reset Parameters
        self.init_weights(weight_init)
        if neuron_parameters['train_neuron_parameters']:
            self.init_neuron_parameters(neuron_parameters)

    def forward(self, x: torch.Tensor):
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
            nn.init.kaiming_normal_(self.conv2d_block.conv2d_layer.weight.data,
                                    a=weight_init['weight_gain'], mode='fan_in')

        elif weight_init['weight_init_dist'] == 'kaiming_uniform_':
            nn.init.kaiming_uniform_(self.conv2d_block.conv2d_layer.weight.data,
                                     a=weight_init['weight_gain'], mode='fan_in')

        elif weight_init['weight_init_dist'] == 'xavier_uniform_':
            nn.init.xavier_uniform_(self.conv2d_block.conv2d_layer.weight.data, gain=weight_init['weight_gain'])

    def init_neuron_parameters(self, neuron_parameters: dict):
        """Method that handles the readout neurons parameters' initialization.

        Parameters
        ----------
        neuron_parameters: dict
            Initialization specifications' dictionary.
        """
        self.conv2d_block.mp_neuron2d_layer.init_neuron_parameters(neuron_parameters)
        self.conv2d_block.mp_neuron2d_layer.clamp_neuron_parameters()


class SpikingResConv2dBlock(nn.Module):
    """Class that implements the SNN spiking residual convolutional block.
    """

    def __init__(self, input_dim: int, output_dim: int, input_channels: int, output_channels: int,
                 kernel_size: Tuple[int, int], stride: Tuple[int, int], padding: Tuple[int, int],
                 dilation: Tuple[int, int], bias: bool, padding_mode: str, nb_steps: int, truncated_bptt_ratio: int,
                 spike_fn: Union[SuperSpike, SigmoidDerivative, PiecewiseLinear, ATan], neuron_model: str,
                 neuron_parameters: dict, weight_init: dict, scale_flag: bool, scale_factor: float, bn_flag: bool,
                 dropout_flag: bool, dropout_p: float, residual_skip_connection_type: str, device, dtype: torch.dtype,
                 layer_index: int) -> None:

        super(SpikingResConv2dBlock, self).__init__()

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
        self.padding_mode = padding_mode

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nb_steps = nb_steps
        self.truncated_bptt_ratio = truncated_bptt_ratio
        self.spike_fn = spike_fn.apply
        self.residual_skip_connection_type = residual_skip_connection_type
        self.layer_index = layer_index

        # Init block_1
        # ------------
        self.conv2d_block_1 = nn.Sequential()

        self.conv2d_block_1.add_module('conv2d_layer',
                                       nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                                 dilation=dilation, bias=self.bias, padding_mode=padding_mode))

        if scale_flag or (not scale_flag and scale_factor != 1):
            self.conv2d_block_1.add_module('scale_layer', ScaleLayer(scale_factor=scale_factor, scale_flag=scale_flag))

        if bn_flag:
            self.conv2d_block_1.add_module('bn_layer', nn.BatchNorm2d(self.output_channels))
            # nn.init.constant_(self.conv2d_block_1.bn_layer.weight.data, val=0.1)
            self.conv2d_block_1.bn_layer.bias = None

        if dropout_flag:
            self.conv2d_block_1.add_module('dropout_layer', nn.Dropout(p=dropout_p))

        if neuron_model == 'lif':
            self.conv2d_block_1.add_module('neuron2d_layer_1',
                                           LIF2d(output_dim, output_channels, truncated_bptt_ratio,
                                                 neuron_parameters['membrane_threshold'],
                                                 neuron_parameters['alpha'], neuron_parameters['beta'],
                                                 neuron_parameters['train_neuron_parameters'], self.spike_fn,
                                                 neuron_parameters['reset_mode'], neuron_parameters['detach_reset'],
                                                 neuron_parameters['recurrent_flag'], neuron_parameters['decay_input'],
                                                 self.device, self.dtype)
                                           )
        elif neuron_model == 'plif':
            self.conv2d_block_1.add_module('neuron2d_layer_1',
                                           PLIF2d(output_dim, output_channels, truncated_bptt_ratio,
                                                  neuron_parameters['membrane_threshold'],
                                                  neuron_parameters['alpha'], neuron_parameters['beta'],
                                                  neuron_parameters['train_neuron_parameters'], self.spike_fn,
                                                  neuron_parameters['reset_mode'], neuron_parameters['detach_reset'],
                                                  neuron_parameters['recurrent_flag'], neuron_parameters['decay_input'],
                                                  self.device, self.dtype)
                                           )
        elif neuron_model == 'klif':
            self.conv2d_block_1.add_module('neuron2d_layer_1',
                                           KLIF2d(output_dim, output_channels, truncated_bptt_ratio,
                                                  neuron_parameters['membrane_threshold'],
                                                  neuron_parameters['alpha'], neuron_parameters['beta'], neuron_parameters['k'],
                                                  neuron_parameters['train_neuron_parameters'], self.spike_fn,
                                                  neuron_parameters['reset_mode'], neuron_parameters['detach_reset'],
                                                  neuron_parameters['recurrent_flag'], neuron_parameters['decay_input'],
                                                  self.device, self.dtype)
                                           )
        elif neuron_model == 'if':
            self.conv2d_block_1.add_module('neuron2d_layer_1',
                                           IF2d(output_dim, output_channels, truncated_bptt_ratio,
                                                neuron_parameters['membrane_threshold'],
                                                neuron_parameters['train_neuron_parameters'], self.spike_fn,
                                                neuron_parameters['reset_mode'], neuron_parameters['detach_reset'],
                                                neuron_parameters['recurrent_flag'], neuron_parameters['decay_input'],
                                                self.device, self.dtype)
                                           )

        # Init block_2
        # ------------
        self.conv2d_block_2 = nn.Sequential()

        self.conv2d_block_2.add_module('conv2d_layer',
                                       nn.Conv2d(in_channels=output_channels, out_channels=output_channels,
                                                 kernel_size=kernel_size, stride=1, padding=padding,
                                                 dilation=dilation, bias=self.bias, padding_mode=padding_mode))

        if scale_flag or (not scale_flag and scale_factor != 1):
            self.conv2d_block_2.add_module('scale_layer', ScaleLayer(scale_factor=scale_factor, scale_flag=scale_flag))

        if bn_flag:
            self.conv2d_block_2.add_module('bn_layer', nn.BatchNorm2d(self.output_channels))
            # nn.init.constant_(self.conv2d_block_2.bn_layer.weight.data, val=0.1)
            self.conv2d_block_2.bn_layer.bias = None

        if dropout_flag:
            self.conv2d_block_2.add_module('dropout_layer', nn.Dropout(p=dropout_p))

        if neuron_model == 'lif':
            self.conv2d_block_2.add_module('neuron2d_layer_2',
                                           LIF2d(output_dim, output_channels, truncated_bptt_ratio,
                                                 neuron_parameters['membrane_threshold'],
                                                 neuron_parameters['alpha'], neuron_parameters['beta'],
                                                 neuron_parameters['train_neuron_parameters'], self.spike_fn,
                                                 neuron_parameters['reset_mode'], neuron_parameters['detach_reset'],
                                                 neuron_parameters['recurrent_flag'], neuron_parameters['decay_input'],
                                                 self.device, self.dtype)
                                           )
        elif neuron_model == 'plif':
            self.conv2d_block_2.add_module('neuron2d_layer_2',
                                           PLIF2d(output_dim, output_channels, truncated_bptt_ratio,
                                                  neuron_parameters['membrane_threshold'],
                                                  neuron_parameters['alpha'], neuron_parameters['beta'],
                                                  neuron_parameters['train_neuron_parameters'], self.spike_fn,
                                                  neuron_parameters['reset_mode'], neuron_parameters['detach_reset'],
                                                  neuron_parameters['recurrent_flag'], neuron_parameters['decay_input'],
                                                  self.device, self.dtype)
                                           )
        elif neuron_model == 'klif':
            self.conv2d_block_2.add_module('neuron2d_layer_2',
                                           KLIF2d(output_dim, output_channels, truncated_bptt_ratio,
                                                  neuron_parameters['membrane_threshold'],
                                                  neuron_parameters['alpha'], neuron_parameters['beta'], neuron_parameters['k'],
                                                  neuron_parameters['train_neuron_parameters'], self.spike_fn,
                                                  neuron_parameters['reset_mode'], neuron_parameters['detach_reset'],
                                                  neuron_parameters['recurrent_flag'], neuron_parameters['decay_input'],
                                                  self.device, self.dtype)
                                           )
        elif neuron_model == 'if':
            self.conv2d_block_2.add_module('neuron2d_layer_2',
                                           IF2d(output_dim, output_channels, truncated_bptt_ratio,
                                                neuron_parameters['membrane_threshold'],
                                                neuron_parameters['train_neuron_parameters'], self.spike_fn,
                                                neuron_parameters['reset_mode'], neuron_parameters['detach_reset'],
                                                neuron_parameters['recurrent_flag'], neuron_parameters['decay_input'],
                                                self.device, self.dtype)
                                           )

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
                                                       dilation=dilation, bias=self.bias, padding_mode=padding_mode))

            if scale_flag or (not scale_flag and scale_factor != 1):
                self.conv2d_block_skp.add_module('scale_layer', ScaleLayer(scale_factor=scale_factor,
                                                                           scale_flag=scale_flag))

            if bn_flag:
                self.conv2d_block_skp.add_module('bn_layer', nn.BatchNorm2d(self.output_channels))
                # nn.init.constant_(self.conv2d_block_skp.bn_layer.weight.data, val=0.1)
                self.conv2d_block_skp.bn_layer.bias = None

            if dropout_flag:
                self.conv2d_block_skp.add_module('dropout_layer', nn.Dropout(p=dropout_p))

            if neuron_model == 'lif':
                self.conv2d_block_skp.add_module('neuron2d_layer_skp',
                                                 LIF2d(output_dim, output_channels, truncated_bptt_ratio,
                                                       neuron_parameters['membrane_threshold'],
                                                       neuron_parameters['alpha'], neuron_parameters['beta'],
                                                       neuron_parameters['train_neuron_parameters'], self.spike_fn,
                                                       neuron_parameters['reset_mode'], neuron_parameters['detach_reset'],
                                                       neuron_parameters['recurrent_flag'],
                                                       neuron_parameters['decay_input'], self.device, self.dtype))
            elif neuron_model == 'plif':
                self.conv2d_block_skp.add_module('neuron2d_layer_skp',
                                                 PLIF2d(output_dim, output_channels, truncated_bptt_ratio,
                                                        neuron_parameters['membrane_threshold'],
                                                        neuron_parameters['alpha'], neuron_parameters['beta'],
                                                        neuron_parameters['train_neuron_parameters'], self.spike_fn,
                                                        neuron_parameters['reset_mode'],
                                                        neuron_parameters['detach_reset'],
                                                        neuron_parameters['recurrent_flag'],
                                                        neuron_parameters['decay_input'], self.device, self.dtype))
            elif neuron_model == 'klif':
                self.conv2d_block_skp.add_module('neuron2d_layer_skp',
                                                 KLIF2d(output_dim, output_channels, truncated_bptt_ratio,
                                                        neuron_parameters['membrane_threshold'],
                                                        neuron_parameters['alpha'], neuron_parameters['beta'], neuron_parameters['k'],
                                                        neuron_parameters['train_neuron_parameters'], self.spike_fn,
                                                        neuron_parameters['reset_mode'],
                                                        neuron_parameters['detach_reset'],
                                                        neuron_parameters['recurrent_flag'],
                                                        neuron_parameters['decay_input'], self.device, self.dtype))
            elif neuron_model == 'if':
                self.conv2d_block_skp.add_module('neuron2d_layer_skp',
                                                 IF2d(output_dim, output_channels, truncated_bptt_ratio,
                                                      neuron_parameters['membrane_threshold'],
                                                      neuron_parameters['train_neuron_parameters'], self.spike_fn,
                                                      neuron_parameters['reset_mode'],
                                                      neuron_parameters['detach_reset'],
                                                      neuron_parameters['recurrent_flag'],
                                                      neuron_parameters['decay_input'], self.device, self.dtype))

        # Reset Parameters
        self.init_weights(weight_init)
        if neuron_parameters['recurrent_flag']:
            self.init_recurrent_weights(weight_init)
        if neuron_parameters['train_neuron_parameters']:
            self.init_neuron_parameters(neuron_parameters)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method that defines the performed computation during the forward pass.
        """

        identity = x

        output_spikes_records_1, membrane_potential_records_1 = self.conv2d_block_1(x)

        output_spikes_records_2, membrane_potential_records_2 = self.conv2d_block_2(output_spikes_records_1)

        if self.init_block_skp:
            identity, membrane_potential_records_skp = self.conv2d_block_skp(identity)

        return self.spk_res_skip_connection(output_spikes_records_2, identity), membrane_potential_records_2

    def spk_res_skip_connection(self, output_spikes_records, identity):
        if self.residual_skip_connection_type == 'add_':
            output_spikes_records = identity + output_spikes_records
        elif self.residual_skip_connection_type == 'and_':
            output_spikes_records = identity * output_spikes_records
        elif self.residual_skip_connection_type == 'iand_':
            output_spikes_records = identity * (1. - output_spikes_records)

        return output_spikes_records

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
                nn.init.normal_(self.conv2d_block_skp.conv2d_layer.weight.data, mean=weight_init['weight_mean'],
                                std=w_std_)

        elif weight_init['weight_init_dist'] == 'uniform_':
            w_bound_ = weight_init['weight_gain'] / np.sqrt(self.input_channels * np.prod(self.kernel_size))
            nn.init.uniform_(self.conv2d_block_1.conv2d_layer.weight.data, a=-w_bound_, b=w_bound_)
            nn.init.uniform_(self.conv2d_block_2.conv2d_layer.weight.data, a=-w_bound_, b=w_bound_)
            if self.init_block_skp:
                nn.init.uniform_(self.conv2d_block_skp.conv2d_layer.weight.data, a=-w_bound_, b=w_bound_)

        elif weight_init['weight_init_dist'] == 'kaiming_normal_':
            nn.init.kaiming_normal_(self.conv2d_block_1.conv2d_layer.weight.data,
                                    a=weight_init['weight_gain'], mode='fan_in')
            nn.init.kaiming_normal_(self.conv2d_block_2.conv2d_layer.weight.data,
                                    a=weight_init['weight_gain'], mode='fan_in')
            if self.init_block_skp:
                nn.init.kaiming_normal_(self.conv2d_block_skp.conv2d_layer.weight.data,
                                        a=weight_init['weight_gain'], mode='fan_in')

        elif weight_init['weight_init_dist'] == 'kaiming_uniform_':
            nn.init.kaiming_uniform_(self.conv2d_block_1.conv2d_layer.weight.data,
                                     a=weight_init['weight_gain'], mode='fan_in')
            nn.init.kaiming_uniform_(self.conv2d_block_2.conv2d_layer.weight.data,
                                     a=weight_init['weight_gain'], mode='fan_in')
            if self.init_block_skp:
                nn.init.kaiming_uniform_(self.conv2d_block_skp.conv2d_layer.weight.data,
                                         a=weight_init['weight_gain'], mode='fan_in')

        elif weight_init['weight_init_dist'] == 'xavier_uniform_':
            nn.init.xavier_uniform_(self.conv2d_block_1.conv2d_layer.weight.data, gain=weight_init['weight_gain'])
            nn.init.xavier_uniform_(self.conv2d_block_2.conv2d_layer.weight.data, gain=weight_init['weight_gain'])
            if self.init_block_skp:
                nn.init.xavier_uniform_(self.conv2d_block_skp.conv2d_layer.weight.data, gain=weight_init['weight_gain'])

    def init_recurrent_weights(self, weight_init: dict):
        """Method that handles the recurrent parameters' initialization.

        Parameters
        ----------
        weight_init: dict
            Initialization specifications' dictionary.
        """
        if weight_init['weight_init_dist'] == 'normal_':
            v_std_ = weight_init['weight_std'] / math.sqrt(self.output_channels)
            nn.init.normal_(self.conv2d_block_1.neuron2d_layer_1.recurrent2d_layer.v.data,
                            mean=weight_init['weight_mean'], std=v_std_)
            nn.init.normal_(self.conv2d_block_2.neuron2d_layer_2.recurrent2d_layer.v.data,
                            mean=weight_init['weight_mean'], std=v_std_)
            if self.init_block_skp:
                nn.init.normal_(self.conv2d_block_skp.neuron2d_layer_skp.recurrent2d_layer.v.data,
                                mean=weight_init['weight_mean'], std=v_std_)

        elif weight_init['weight_init_dist'] == 'uniform_':
            v_bound_ = weight_init['weight_gain'] / math.sqrt(self.output_channels)
            nn.init.uniform_(self.conv2d_block_1.neuron2d_layer_1.recurrent2d_layer.v.data, a=-v_bound_, b=v_bound_)
            nn.init.uniform_(self.conv2d_block_2.neuron2d_layer_2.recurrent2d_layer.v.data, a=-v_bound_, b=v_bound_)
            if self.init_block_skp:
                nn.init.uniform_(self.conv2d_block_skp.neuron2d_layer_skp.recurrent2d_layer.v.data,
                                 a=-v_bound_, b=v_bound_)

        elif weight_init['weight_init_dist'] == 'kaiming_normal_':
            nn.init.kaiming_normal_(self.conv2d_block_1.neuron2d_layer_1.recurrent2d_layer.v.data,
                                    a=weight_init['weight_gain'], mode='fan_out')
            nn.init.kaiming_normal_(self.conv2d_block_2.neuron2d_layer_2.recurrent2d_layer.v.data,
                                    a=weight_init['weight_gain'], mode='fan_out')
            if self.init_block_skp:
                nn.init.kaiming_normal_(self.conv2d_block_skp.neuron2d_layer_skp.recurrent2d_layer.v.data,
                                        a=weight_init['weight_gain'], mode='fan_out')

        elif weight_init['weight_init_dist'] == 'kaiming_uniform_':
            nn.init.kaiming_uniform_(self.conv2d_block_1.neuron2d_layer_1.recurrent2d_layer.v.data,
                                     a=weight_init['weight_gain'], mode='fan_out')
            nn.init.kaiming_uniform_(self.conv2d_block_2.neuron2d_layer_2.recurrent2d_layer.v.data,
                                     a=weight_init['weight_gain'], mode='fan_out')
            if self.init_block_skp:
                nn.init.kaiming_uniform_(self.conv2d_block_skp.neuron2d_layer_skp.recurrent2d_layer.v.data,
                                         a=weight_init['weight_gain'], mode='fan_out')

        elif weight_init['weight_init_dist'] == 'xavier_uniform_':
            nn.init.xavier_uniform_(self.conv2d_block_1.neuron2d_layer_1.recurrent2d_layer.v.data,
                                    gain=weight_init['weight_gain'])
            nn.init.xavier_uniform_(self.conv2d_block_2.neuron2d_layer_2.recurrent2d_layer.v.data,
                                    gain=weight_init['weight_gain'])
            if self.init_block_skp:
                nn.init.xavier_uniform_(self.conv2d_block_skp.neuron2d_layer_skp.recurrent2d_layer.v.data,
                                        gain=weight_init['weight_gain'])

    def init_neuron_parameters(self, neuron_parameters: dict):
        """Method that handles the spiking neurons parameters' initialization.

        Parameters
        ----------
        neuron_parameters: dict
            Initialization specifications' dictionary.
        """
        self.conv2d_block_1.neuron2d_layer_1.init_neuron_parameters(neuron_parameters)
        self.conv2d_block_1.neuron2d_layer_1.clamp_neuron_parameters()

        self.conv2d_block_2.neuron2d_layer_2.init_neuron_parameters(neuron_parameters)
        self.conv2d_block_2.neuron2d_layer_2.clamp_neuron_parameters()
        if self.init_block_skp:
            self.conv2d_block_skp.neuron2d_layer_skp.init_neuron_parameters(neuron_parameters)
            self.conv2d_block_skp.neuron2d_layer_skp.clamp_neuron_parameters()
