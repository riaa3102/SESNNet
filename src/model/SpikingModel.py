"""
File:
    model/SpikingModel.py

Description:
    Defines the SNNBase, FCSNN, CSNN, UNetSNN and ResBottleneckUNetSNN class.
"""

from typing import Optional, Union, List, Tuple
import numpy as np
import math
import torch
import torch.nn as nn
from src.model.constants import Conv2dParameters, UnetConv2dParameters
from src.model.utils import set_conv2d_parameters, set_pooling2d_parameters, set_residual_conv2d_parameters, \
    compute_conv_out_shape
from src.model.SpikingBlock import SpikingFCBlock, ReadoutFCBlock, SpikingConv2dBlock, ReadoutConv2dBlock, \
    SpikingResConv2dBlock
from src.model.SpikingLayer import Upsampling2d, LIF1d, LI1d, LIF2d, LI2d, PLIF2d, PLI2d, IF2d, I2d
from src.model.SurrogateGradient import SuperSpike, SigmoidDerivative, PiecewiseLinear, ATan


class SNNBase(nn.Module):
    """Class that implements the base SNN model architecture as a set of sequential spiking layers.
    """

    def __init__(self, layers_list: nn.ModuleList, train_neuron_parameters: bool = False) -> None:
        """Method that initializes the class instance.

        Parameters
        ----------
        layers_list: nn.ModuleList
            List of spiking layers that is used to build the SNN model.
        train_neuron_parameters: bool
            Boolean that indicates weather to train neuron parameters.
        """

        super(SNNBase, self).__init__()

        self.snn_layers = layers_list
        self.nb_layers = len(layers_list)
        self.train_neuron_parameters = train_neuron_parameters
        self.layers_name = self._layers_name()
        self._save_mem = False
        self.spk_rec = []
        self.mem_rec = []

    @property
    def save_mem(self) -> bool:
        return self._save_mem

    @save_mem.setter
    def save_mem(self, value: bool) -> None:
        self._save_mem = value
        for snn_layer in self.snn_layers:
            if isinstance(snn_layer, SpikingFCBlock):
                snn_layer.fc_block.neuron1d_layer.save_mem = value
            elif isinstance(snn_layer, SpikingConv2dBlock):
                snn_layer.conv2d_block.neuron2d_layer.save_mem = value
            elif isinstance(snn_layer, SpikingResConv2dBlock):
                snn_layer.conv2d_block_1.neuron2d_layer_1.save_mem = value
                snn_layer.conv2d_block_2.neuron2d_layer_2.save_mem = value
                if snn_layer.init_block_skp:
                    snn_layer.conv2d_block_skp.neuron2d_layer_layer_skp.save_mem = value

    def init_state(self, batch_size: int) -> None:
        """Method that initializes the layers state to tensors of zeros.

        Parameters
        ----------
        batch_size: int
            Data batch size.
        """
        for snn_layer in self.snn_layers:
            for snn_layer_i in snn_layer.children():
                if isinstance(snn_layer_i, nn.Sequential):
                    for snn_layer_i_j in snn_layer_i.children():
                        if isinstance(snn_layer_i_j, (LIF2d, LI2d, PLIF2d, PLI2d, IF2d, I2d)):
                            for states_dict_k in snn_layer_i_j.states_dict:
                                snn_layer_i_j.states_dict[states_dict_k] = torch.zeros(
                                    (batch_size, snn_layer_i_j.output_channels, snn_layer_i_j.output_dim),
                                    device=snn_layer_i_j.device, dtype=snn_layer_i_j.dtype)
                        elif isinstance(snn_layer_i_j, (LIF1d, LI1d)):
                            for states_dict_k in snn_layer_i_j.states_dict:
                                snn_layer_i_j.states_dict[states_dict_k] = torch.zeros(
                                    (batch_size, snn_layer_i_j.output_dim),
                                    device=snn_layer_i_j.device, dtype=snn_layer_i_j.dtype)

    def init_state_none(self) -> None:
        """Method that initializes the layers state to None value.
        """
        for snn_layer in self.snn_layers:
            for snn_layer_i in snn_layer.children():
                if isinstance(snn_layer_i, nn.Sequential):
                    for snn_layer_i_j in snn_layer_i.children():
                        if isinstance(snn_layer_i_j, (LIF1d, LI1d, LIF2d, LI2d, PLIF2d, PLI2d, IF2d, I2d)):
                            for states_dict_k in snn_layer_i_j.states_dict:
                                snn_layer_i_j.states_dict[states_dict_k] = None

    def init_rec(self) -> None:
        """Method that initializes the lists of output of SNN layers.
        """
        self.spk_rec = []
        self.mem_rec = []

    def _detach_rec(self) -> None:
        """Method that detaches SNN output records from the current computational graph.
        """
        if self._save_mem:
            for i in range(len(self.spk_rec)):
                self.spk_rec[i] = self.spk_rec[i].clone().detach().cpu().numpy()
        else:
            for i in range(len(self.spk_rec)):
                self.spk_rec[i] = None

    def _layers_name(self) -> List:
        """Method that creates list of SNN layers name.
        """
        snn_layer_name_list = []
        for snn_layer_i in self.snn_layers:
            snn_layer_name = snn_layer_i.__class__.__name__
            for snn_layer_i_j in snn_layer_i.children():
                if isinstance(snn_layer_i_j, nn.Sequential):
                    snn_layer_name_list.append(snn_layer_name)
        return snn_layer_name_list

    def update_nb_steps(self, nb_steps_bin: int, nb_steps: int) -> None:
        """Method that initializes the layers state to tensors of zeros.

        Parameters
        ----------
        nb_steps_bin: int
            Number of time steps per segment.
        nb_steps: int
            Number of data time steps.
        """
        for snn_layer in self.snn_layers:
            for snn_layer_i in snn_layer.children():
                if isinstance(snn_layer_i, nn.Sequential):
                    for snn_layer_i_j in snn_layer_i.children():
                        if isinstance(snn_layer_i_j, Upsampling2d):
                            scale_factor = math.ceil(nb_steps_bin / snn_layer_i_j.nb_steps)
                            snn_layer_i_j.nb_steps = math.ceil(nb_steps / scale_factor)

    def clamp_neuron_parameters(self) -> None:
        """Method that clamps the parameters of SNN neurons.
        """
        for snn_layer in self.snn_layers:
            if isinstance(snn_layer, SpikingFCBlock):
                snn_layer.fc_block.neuron1d_layer.clamp_neuron_parameters()
            elif isinstance(snn_layer, ReadoutFCBlock):
                snn_layer.fc_block.mp_neuron1d_layer.clamp_neuron_parameters()
            elif isinstance(snn_layer, SpikingConv2dBlock):
                snn_layer.conv2d_block.neuron2d_layer.clamp_neuron_parameters()
            elif isinstance(snn_layer, SpikingResConv2dBlock):
                snn_layer.conv2d_block_1.neuron2d_layer_1.clamp_neuron_parameters()
                snn_layer.conv2d_block_2.neuron2d_layer_2.clamp_neuron_parameters()
                if snn_layer.init_block_skp:
                    snn_layer.conv2d_block_skp.neuron2d_layer_layer_skp.clamp_neuron_parameters()
            elif isinstance(snn_layer, ReadoutConv2dBlock):
                snn_layer.conv2d_block.mp_neuron2d_layer.clamp_neuron_parameters()


class FCSNN(SNNBase):
    """Class that implements the SNN model architecture with predefined list of fully connected layers.
    """

    def __init__(self, input_dim: int, hidden_dim_list: list, output_dim: int, nb_steps: int, truncated_bptt_ratio: int,
                 spike_fn: Union[SuperSpike, SigmoidDerivative, PiecewiseLinear, ATan], neuron_model: str,
                 neuron_parameters: dict, weight_init: dict, scale_flag: bool = True, scale_factor: float = 1.,
                 bn_flag: bool = False, dropout_flag: bool = False, dropout_p: float = 0.5,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float) -> None:

        nb_hidden_layers = len(hidden_dim_list) - 1

        # Init
        layer_index = 0
        layers_list = nn.ModuleList()

        # ------------------------------------------------- Input Layer ------------------------------------------------
        scale_factor_ = 4.
        # --------------------------
        layers_list.append(
            SpikingFCBlock(input_dim, hidden_dim_list[0], nb_steps, truncated_bptt_ratio,
                           spike_fn, neuron_model, neuron_parameters, weight_init,
                           scale_flag, scale_factor_, bn_flag, dropout_flag, dropout_p,
                           device, dtype, layer_index))

        # ---------------------------------------------- Hidden Layer(s) -----------------------------------------------
        for i in range(nb_hidden_layers):
            layer_index += 1
            layers_list.append(
                SpikingFCBlock(hidden_dim_list[i], hidden_dim_list[i + 1], nb_steps, truncated_bptt_ratio,
                               spike_fn, neuron_model, neuron_parameters, weight_init,
                               scale_flag, scale_factor, bn_flag, dropout_flag, dropout_p,
                               device, dtype, layer_index))

        # ----------------------------------------------- Readout Layer ------------------------------------------------
        layer_index += 1
        layers_list.append(
            ReadoutFCBlock(hidden_dim_list[-1], output_dim, nb_steps, truncated_bptt_ratio,
                           spike_fn, neuron_model, neuron_parameters,
                           weight_init,
                           scale_flag, scale_factor, device, dtype, layer_index))

        super(FCSNN, self).__init__(layers_list, neuron_parameters['train_neuron_parameters'])

    def forward(self, spk: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Method that defines the performed computation during the forward pass.
        """

        self.spk_rec.append(spk)
        output = []

        if spk.dim() == 4:
            spk = spk.view(spk.shape[0], spk.shape[-2], spk.shape[-1])

        for i, snn_layer in enumerate(self.snn_layers):

            if isinstance(snn_layer, SpikingFCBlock):
                spk, mem = snn_layer(spk)

                if self.save_mem:
                    self.mem_rec.append(mem.clone().detach().cpu().numpy())
                self.spk_rec.append(spk)

            elif isinstance(snn_layer, ReadoutFCBlock):
                mem = snn_layer(spk)

                if self.save_mem:
                    self.mem_rec.append(mem.clone().detach().cpu().numpy())

        self._detach_rec()

        return mem, output


class CSNN(SNNBase):
    """Class that implements the SNN model architecture with predefined list of convolutional layers.
    """

    def __init__(self, input_dim: int, hidden_channels_list: list, output_dim: int, kernel_size: tuple, stride: tuple,
                 padding: tuple, dilation: tuple, bias: bool, padding_mode: str, pooling_flag: bool, pooling_type: str,
                 use_same_layer: bool, nb_steps: int, truncated_bptt_ratio: int,
                 spike_fn: Union[SuperSpike, SigmoidDerivative, PiecewiseLinear, ATan],
                 neuron_model: str, neuron_parameters: dict, weight_init: dict, upsample_mode: str = 'bilinear',
                 scale_flag: bool = True, scale_factor: float = 1., bn_flag: bool = False,
                 dropout_flag: bool = False, dropout_p: float = 0.5,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float) -> None:

        # ------------------------------------------------ Init Params -------------------------------------------------

        hidden_channels_list_ = hidden_channels_list

        hidden_channels_list, kernel_size_list, stride_list, padding_list, dilation_list = set_conv2d_parameters(
            UnetConv2dParameters, hidden_channels_list, kernel_size, stride, padding, dilation, use_same_layer,
            pooling_flag)

        if pooling_flag:
            pooling_stride_list, pooling_padding_list = set_pooling2d_parameters(UnetConv2dParameters,
                                                                                 hidden_channels_list_, kernel_size,
                                                                                 stride, use_same_layer)
        else:
            pooling_stride_list = [None for i in range(len(hidden_channels_list))]
            pooling_padding_list = [None for i in range(len(hidden_channels_list))]

        hidden_dim = []

        # ------------------------------------------------
        nb_encoder_layers = len(hidden_channels_list) - 1

        # Init
        layers_list = nn.ModuleList()
        input_channels = 1
        output_channels = 1

        # ------------------------------------------------ Input Layer -------------------------------------------------
        layer_index = 0
        upsample_flag = False
        dropout_flag_ = False
        # --------------------------
        # Compute hidden layer output
        if not pooling_flag:
            hidden_output_dim, hidden_nb_steps = compute_conv_out_shape(input_dim, nb_steps, kernel_size_list[0],
                                                                        stride_list[0], padding_list[0],
                                                                        dilation_list[0])
        else:
            hidden_output_dim, hidden_nb_steps = compute_conv_out_shape(input_dim, nb_steps, kernel_size_list[0],
                                                                        pooling_stride_list[0], pooling_padding_list[0],
                                                                        dilation_list[0])
        hidden_dim.append((hidden_output_dim, hidden_nb_steps))
        # --------------------------
        layers_list.append(
            SpikingConv2dBlock(input_dim, hidden_output_dim, input_channels, hidden_channels_list[0],
                               kernel_size_list[0], stride_list[0], padding_list[0], dilation_list[0], bias,
                               padding_mode,
                               pooling_flag, pooling_type, pooling_stride_list[0], pooling_padding_list[0],
                               hidden_nb_steps, truncated_bptt_ratio, spike_fn, neuron_model, neuron_parameters,
                               weight_init, upsample_flag, upsample_mode,
                               scale_flag, scale_factor, bn_flag, dropout_flag_, dropout_p,
                               device, dtype, layer_index))

        # ---------------------------------------------- Hidden Layer(s) -----------------------------------------------
        for i in range(nb_encoder_layers):
            layer_index += 1
            # --------------------------
            hidden_input_dim = hidden_output_dim
            if not pooling_flag:
                hidden_output_dim, hidden_nb_steps = compute_conv_out_shape(hidden_input_dim, hidden_nb_steps,
                                                                            kernel_size_list[i + 1], stride_list[i + 1],
                                                                            padding_list[i + 1], dilation_list[i + 1])
            else:
                hidden_output_dim, hidden_nb_steps = compute_conv_out_shape(hidden_input_dim, hidden_nb_steps,
                                                                            kernel_size_list[i + 1],
                                                                            pooling_stride_list[i + 1],
                                                                            pooling_padding_list[i + 1],
                                                                            dilation_list[i + 1])
            hidden_dim.append((hidden_output_dim, hidden_nb_steps))
            # --------------------------
            layers_list.append(
                SpikingConv2dBlock(hidden_input_dim, hidden_output_dim, hidden_channels_list[i],
                                   hidden_channels_list[i + 1], kernel_size_list[i + 1], stride_list[i + 1],
                                   padding_list[i + 1], dilation_list[i + 1], bias, padding_mode,
                                   pooling_flag, pooling_type, pooling_stride_list[i + 1],
                                   pooling_padding_list[i + 1],
                                   hidden_nb_steps, truncated_bptt_ratio,
                                   spike_fn, neuron_model, neuron_parameters, weight_init,
                                   upsample_flag, upsample_mode,
                                   scale_flag, scale_factor, bn_flag, dropout_flag_, dropout_p,
                                   device, dtype, layer_index))

        # ---------------------------------------------- Hidden Layer(s) -----------------------------------------------
        hidden_channels_list.reverse()
        kernel_size_list.reverse()
        if pooling_flag:
            stride_list = pooling_stride_list
        stride_list.reverse()
        stride_list_ = [(1, 1) for i in range(len(hidden_channels_list))]
        padding_list.reverse()
        hidden_dim.reverse()
        # --------------------------
        pooling_flag = False
        pooling_stride = None
        pooling_padding = None
        # --------------------------
        for i in range(nb_encoder_layers):
            # --------------------------
            layer_index += 1
            # --------------------------
            hidden_input_dim = hidden_output_dim
            # --------------------------
            upsample_flag_ = upsample_flag
            if stride_list[i][0] != 1:
                upsample_flag_ = True
                hidden_input_dim *= stride_list[i][0]
            if stride_list[i][1] != 1:
                upsample_flag_ = True
            # --------------------------
            hidden_output_dim, hidden_nb_steps = hidden_dim[i + 1]
            # --------------------------
            if i > 2:
                dropout_flag = False
            # --------------------------
            layers_list.append(
                SpikingConv2dBlock(hidden_input_dim, hidden_output_dim, hidden_channels_list[i],
                                   hidden_channels_list[i + 1], kernel_size_list[i], stride_list_[i], padding_list[i],
                                   dilation_list[i], bias, padding_mode,
                                   pooling_flag, pooling_type, pooling_stride, pooling_padding,
                                   hidden_nb_steps, truncated_bptt_ratio, spike_fn,
                                   neuron_model, neuron_parameters, weight_init,
                                   upsample_flag_, upsample_mode,
                                   scale_flag, scale_factor, bn_flag, dropout_flag, dropout_p,
                                   device, dtype, layer_index))
        # ----------------------------------------------- Readout Layer ------------------------------------------------
        upsample_flag = True
        layer_index += 1
        # --------------------------
        layers_list.append(
            ReadoutConv2dBlock(output_dim, output_dim, hidden_channels_list[-1],
                               output_channels, kernel_size_list[-1], stride_list_[-1], padding_list[-1],
                               dilation_list[-1], bias, padding_mode,
                               nb_steps, truncated_bptt_ratio, neuron_model, neuron_parameters,
                               weight_init, upsample_flag, upsample_mode, scale_flag,
                               scale_factor, device, dtype, layer_index))

        # --------------------------
        hidden_channels_list.reverse()
        # --------------------------

        super(CSNN, self).__init__(layers_list, neuron_parameters['train_neuron_parameters'])

    def forward(self, spk: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Method that defines the performed computation during the forward pass.
        """

        self.spk_rec.append(spk)
        output = []

        for i, snn_layer in enumerate(self.snn_layers):

            if isinstance(snn_layer, SpikingConv2dBlock):
                spk, mem = snn_layer(spk)

                if self.save_mem:
                    self.mem_rec.append(mem.clone().detach().cpu().numpy())
                self.spk_rec.append(spk)

            elif isinstance(snn_layer, ReadoutConv2dBlock):
                mem = snn_layer(spk)

                if self.save_mem:
                    self.mem_rec.append(mem.clone().detach().cpu().numpy())

        self._detach_rec()

        return mem, output


class UNetSNN(SNNBase):
    """Class that implements the SNN model architecture with predefined list of convolutional layers.
    """

    def __init__(self, input_dim: int, hidden_channels_list: list, output_dim: int, kernel_size: tuple, stride: tuple,
                 padding: tuple, dilation: tuple, bias: bool, padding_mode: str, pooling_flag: bool, pooling_type: str,
                 use_same_layer: bool, nb_steps: int, truncated_bptt_ratio: int,
                 spike_fn: Union[SuperSpike, SigmoidDerivative, PiecewiseLinear, ATan], neuron_model: str,
                 neuron_parameters: dict, weight_init: dict, upsample_mode: str = 'bilinear', scale_flag: bool = True,
                 scale_factor: float = 1., bn_flag: bool = False, dropout_flag: bool = False, dropout_p: float = 0.5,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float, skip_connection_type: str = 'cat_',
                 use_intermediate_output: bool = False) -> None:

        # ------------------------------------------------ Init Params -------------------------------------------------

        hidden_channels_list_ = hidden_channels_list

        hidden_channels_list, kernel_size_list, stride_list, padding_list, dilation_list = set_conv2d_parameters(
            UnetConv2dParameters, hidden_channels_list, kernel_size, stride, padding, dilation, use_same_layer,
            pooling_flag)

        if pooling_flag:
            pooling_stride_list, pooling_padding_list = set_pooling2d_parameters(UnetConv2dParameters,
                                                                                 hidden_channels_list_, kernel_size,
                                                                                 stride, use_same_layer)
        else:
            pooling_stride_list = [None for i in range(len(hidden_channels_list))]
            pooling_padding_list = [None for i in range(len(hidden_channels_list))]

        hidden_dim = []

        # ------------------------------------------------

        nb_encoder_layers = len(hidden_channels_list) - 1

        # Init
        layers_list = nn.ModuleList()
        input_channels = 1
        output_channels = 1

        # ------------------------------------------------ Input Layer -------------------------------------------------
        layer_index = 0
        upsample_flag = False
        dropout_flag_ = False
        # --------------------------
        # Compute hidden layer output
        if not pooling_flag:
            hidden_output_dim, hidden_nb_steps = compute_conv_out_shape(input_dim, nb_steps, kernel_size_list[0],
                                                                        stride_list[0], padding_list[0],
                                                                        dilation_list[0])
        else:
            hidden_output_dim, hidden_nb_steps = compute_conv_out_shape(input_dim, nb_steps, kernel_size_list[0],
                                                                        pooling_stride_list[0], pooling_padding_list[0],
                                                                        dilation_list[0])
        hidden_dim.append((hidden_output_dim, hidden_nb_steps))
        # --------------------------
        layers_list.append(
            SpikingConv2dBlock(input_dim, hidden_output_dim, input_channels, hidden_channels_list[0],
                               kernel_size_list[0], stride_list[0], padding_list[0], dilation_list[0], bias, padding_mode,
                               pooling_flag, pooling_type, pooling_stride_list[0], pooling_padding_list[0],
                               hidden_nb_steps, truncated_bptt_ratio, spike_fn, neuron_model, neuron_parameters,
                               weight_init, upsample_flag, upsample_mode,
                               scale_flag, scale_factor, bn_flag, dropout_flag_, dropout_p,
                               device, dtype, layer_index))

        # ----------------------------------------------- Encoder block ------------------------------------------------
        for i in range(nb_encoder_layers):
            # --------------------------
            layer_index += 1
            # --------------------------
            hidden_input_dim = hidden_output_dim
            if not pooling_flag:
                hidden_output_dim, hidden_nb_steps = compute_conv_out_shape(hidden_input_dim, hidden_nb_steps,
                                                                            kernel_size_list[i + 1], stride_list[i + 1],
                                                                            padding_list[i + 1], dilation_list[i + 1])
            else:
                hidden_output_dim, hidden_nb_steps = compute_conv_out_shape(hidden_input_dim, hidden_nb_steps,
                                                                            kernel_size_list[i + 1],
                                                                            pooling_stride_list[i + 1],
                                                                            pooling_padding_list[i + 1],
                                                                            dilation_list[i + 1])
            hidden_dim.append((hidden_output_dim, hidden_nb_steps))
            # --------------------------
            layers_list.append(
                SpikingConv2dBlock(hidden_input_dim, hidden_output_dim, hidden_channels_list[i],
                                   hidden_channels_list[i + 1], kernel_size_list[i+1], stride_list[i+1],
                                   padding_list[i+1], dilation_list[i+1], bias, padding_mode,
                                   pooling_flag, pooling_type, pooling_stride_list[i+1],
                                   pooling_padding_list[i+1],
                                   hidden_nb_steps, truncated_bptt_ratio,
                                   spike_fn, neuron_model, neuron_parameters, weight_init,
                                   upsample_flag, upsample_mode,
                                   scale_flag, scale_factor, bn_flag, dropout_flag_, dropout_p,
                                   device, dtype, layer_index))

        # ----------------------------------------------- Decoder block ------------------------------------------------
        hidden_channels_list.reverse()
        kernel_size_list.reverse()
        if pooling_flag:
            stride_list = pooling_stride_list
        stride_list.reverse()
        stride_list_ = [(1, 1) for i in range(len(hidden_channels_list))]
        padding_list.reverse()
        hidden_dim.reverse()
        # --------------------------
        pooling_flag = False
        pooling_stride = None
        pooling_padding = None
        # --------------------------
        for i in range(nb_encoder_layers):
            # --------------------------
            layer_index += 1
            # --------------------------
            hidden_input_dim = hidden_output_dim
            # --------------------------
            upsample_flag_ = upsample_flag
            if stride_list[i][0] != 1:
                upsample_flag_ = True
                hidden_input_dim *= stride_list[i][0]
            if stride_list[i][1] != 1:
                upsample_flag_ = True
            # --------------------------
            hidden_output_dim, hidden_nb_steps = hidden_dim[i + 1]
            # --------------------------
            if i == 0:
                cat_scale_factor = 1
            elif skip_connection_type == 'cat_':
                cat_scale_factor = 2
            # --------------------------
            if i > 2:
                dropout_flag = False
            # --------------------------
            layers_list.append(
                SpikingConv2dBlock(hidden_input_dim, hidden_output_dim, cat_scale_factor * hidden_channels_list[i],
                                   hidden_channels_list[i + 1], kernel_size_list[i], stride_list_[i], padding_list[i],
                                   dilation_list[i], bias, padding_mode,
                                   pooling_flag, pooling_type, pooling_stride, pooling_padding,
                                   hidden_nb_steps, truncated_bptt_ratio, spike_fn,
                                   neuron_model, neuron_parameters, weight_init,
                                   upsample_flag_, upsample_mode,
                                   scale_flag, scale_factor, bn_flag, dropout_flag, dropout_p,
                                   device, dtype, layer_index))

            # ---------------------------------------- Intermediate Readout Layer ------------------------------------------
            if i < (nb_encoder_layers - 1) and use_intermediate_output:
                upsample_flag = True
                layer_index += 1
                # --------------------------
                if skip_connection_type == 'cat_':
                    cat_scale_factor = 2
                # --------------------------
                layers_list.append(
                    ReadoutConv2dBlock(output_dim, output_dim, cat_scale_factor * hidden_channels_list[i + 1],
                                       output_channels, kernel_size_list[-1], stride_list_[-1], padding_list[-1],
                                       dilation_list[-1], bias, padding_mode,
                                       nb_steps, truncated_bptt_ratio, neuron_model, neuron_parameters,
                                       weight_init, upsample_flag, upsample_mode,
                                       scale_flag, scale_factor, device, dtype, layer_index))

        # ----------------------------------------------- Readout Layer ------------------------------------------------
        upsample_flag = True
        layer_index += 1
        # --------------------------
        if skip_connection_type == 'cat_':
            cat_scale_factor = 2
        # --------------------------
        layers_list.append(
            ReadoutConv2dBlock(output_dim, output_dim, cat_scale_factor * hidden_channels_list[-1],
                               output_channels, kernel_size_list[-1], stride_list_[-1], padding_list[-1],
                               dilation_list[-1], bias, padding_mode,
                               nb_steps, truncated_bptt_ratio, neuron_model, neuron_parameters,
                               weight_init, upsample_flag, upsample_mode, scale_flag,
                               scale_factor, device, dtype, layer_index))

        # --------------------------
        hidden_channels_list.reverse()
        # --------------------------

        super(UNetSNN, self).__init__(layers_list, neuron_parameters['train_neuron_parameters'])

        self.skip_connection_type = skip_connection_type
        self.nb_skip_connection = nb_encoder_layers
        self.use_intermediate_output = use_intermediate_output

    def forward(self, spk: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Method that defines the performed computation during the forward pass.
        """

        self.spk_rec.append(spk)
        output = []

        skip_connection_k = 0
        for i, snn_layer in enumerate(self.snn_layers):

            if isinstance(snn_layer, SpikingConv2dBlock):

                if i <= (self.nb_skip_connection + 1):
                    spk, mem = snn_layer(spk)
                else:
                    if not self.use_intermediate_output:
                        spk_rec_k_index = self.nb_skip_connection - skip_connection_k
                        spk = self.spk_skip_connection(spk, spk_rec_k_index)
                        self.spk_rec[-1] = spk
                        skip_connection_k += 1

                    spk, mem = snn_layer(spk)

                if self.save_mem:
                    self.mem_rec.append(mem.clone().detach().cpu().numpy())
                self.spk_rec.append(spk)

            elif isinstance(snn_layer, ReadoutConv2dBlock):

                if self.use_intermediate_output:

                    spk_rec_k_index = self.nb_skip_connection - skip_connection_k
                    spk = self.spk_skip_connection(spk, spk_rec_k_index)
                    self.spk_rec[-1] = spk
                    skip_connection_k += 1

                    mem = snn_layer(spk)

                    if i < (len(self.snn_layers) - 1):
                        output.append(mem)
                    else:
                        if self.save_mem:
                            self.mem_rec.append(mem.clone().detach().cpu().numpy())

                else:
                    spk = self.spk_skip_connection(spk, 1)
                    self.spk_rec[-1] = spk

                    mem = snn_layer(spk)

                    if self.save_mem:
                        self.mem_rec.append(mem.clone().detach().cpu().numpy())

        self._detach_rec()

        return mem, output

    def spk_skip_connection(self, spk: torch.Tensor, spk_rec_k_index: int) -> torch.Tensor:
        """Method that creates spiking skip connections output.

        Parameters
        ----------
        spk: torch.Tensor
            Input spike tensor of corresponding layer.
        spk_rec_k_index: int
            Spike tensor index within spike records.
        """
        if self.skip_connection_type == 'cat_':
            spk = torch.cat((spk, self.spk_rec[spk_rec_k_index]), 1)
        elif self.skip_connection_type == 'add_':
            spk = torch.add(spk, self.spk_rec[spk_rec_k_index])
        return spk


class ResBottleneckUNetSNN(SNNBase):
    """Class that implements the SNN model architecture with predefined list of convolutional layers.
    """

    def __init__(self, input_dim: int, hidden_channels_list: list, output_dim: int, kernel_size: tuple, stride: tuple,
                 padding: tuple, dilation: tuple, bias: bool, padding_mode: str, pooling_flag: bool, pooling_type: str,
                 use_same_layer: bool, nb_steps: int, truncated_bptt_ratio: int,
                 spike_fn: Union[SuperSpike, SigmoidDerivative, PiecewiseLinear, ATan], neuron_model: str,
                 neuron_parameters: dict, weight_init: dict, scale_flag: bool = True, upsample_mode: str = 'bilinear',
                 scale_factor: float = 1., bn_flag: bool = False, dropout_flag: bool = False, dropout_p: float = 0.5,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float, skip_connection_type: str = 'cat_',
                 nb_residual_block: int = 1, residual_skip_connection_type: str = 'add_',
                 use_intermediate_output: bool = False) -> None:

        # ------------------------------------------------ Init Params -------------------------------------------------

        hidden_channels_list_ = hidden_channels_list

        hidden_channels_list, kernel_size_list, stride_list, padding_list, dilation_list = set_conv2d_parameters(
            UnetConv2dParameters, hidden_channels_list, kernel_size, stride, padding, dilation, use_same_layer,
            pooling_flag)

        if pooling_flag:
            pooling_stride_list, pooling_padding_list = set_pooling2d_parameters(UnetConv2dParameters,
                                                                                 hidden_channels_list_, kernel_size,
                                                                                 stride, use_same_layer)
        else:
            pooling_stride_list = [None for i in range(len(hidden_channels_list))]
            pooling_padding_list = [None for i in range(len(hidden_channels_list))]

        residual_kernel_size, residual_stride, residual_padding, residual_dilation = set_residual_conv2d_parameters()

        hidden_dim = []

        # ------------------------------------------------

        nb_encoder_layers = len(hidden_channels_list) - 1

        # Init
        layers_list = nn.ModuleList()
        input_channels = 1
        output_channels = 1

        # ------------------------------------------------ Input Layer -------------------------------------------------
        layer_index = 0
        upsample_flag = False
        dropout_flag_ = False
        # --------------------------
        # Compute hidden layer output
        if not pooling_flag:
            hidden_output_dim, hidden_nb_steps = compute_conv_out_shape(input_dim, nb_steps, kernel_size_list[0],
                                                                        stride_list[0], padding_list[0],
                                                                        dilation_list[0])
        else:
            hidden_output_dim, hidden_nb_steps = compute_conv_out_shape(input_dim, nb_steps, kernel_size_list[0],
                                                                        pooling_stride_list[0], pooling_padding_list[0],
                                                                        dilation_list[0])
        hidden_dim.append((hidden_output_dim, hidden_nb_steps))
        # --------------------------
        layers_list.append(
            SpikingConv2dBlock(input_dim, hidden_output_dim, input_channels, hidden_channels_list[0],
                               kernel_size_list[0], stride_list[0], padding_list[0], dilation_list[0], bias, padding_mode,
                               pooling_flag, pooling_type, pooling_stride_list[0], pooling_padding_list[0],
                               hidden_nb_steps, truncated_bptt_ratio, spike_fn, neuron_model, neuron_parameters,
                               weight_init, upsample_flag, upsample_mode,
                               scale_flag, scale_factor, bn_flag, dropout_flag_, dropout_p,
                               device, dtype, layer_index))

        # ----------------------------------------------- Encoder block ------------------------------------------------
        for i in range(nb_encoder_layers):
            # --------------------------
            layer_index += 1
            # --------------------------
            hidden_input_dim = hidden_output_dim
            if not pooling_flag:
                hidden_output_dim, hidden_nb_steps = compute_conv_out_shape(hidden_input_dim, hidden_nb_steps,
                                                                            kernel_size_list[i + 1], stride_list[i + 1],
                                                                            padding_list[i + 1], dilation_list[i + 1])
            else:
                hidden_output_dim, hidden_nb_steps = compute_conv_out_shape(hidden_input_dim, hidden_nb_steps,
                                                                            kernel_size_list[i + 1],
                                                                            pooling_stride_list[i + 1],
                                                                            pooling_padding_list[i + 1],
                                                                            dilation_list[i + 1])
            hidden_dim.append((hidden_output_dim, hidden_nb_steps))
            # -------------------------------------------------------------
            layers_list.append(
                SpikingConv2dBlock(hidden_input_dim, hidden_output_dim, hidden_channels_list[i],
                                   hidden_channels_list[i + 1], kernel_size_list[i + 1], stride_list[i + 1],
                                   padding_list[i + 1], dilation_list[i + 1], bias, padding_mode,
                                   pooling_flag, pooling_type, pooling_stride_list[i + 1], pooling_padding_list[i + 1],
                                   hidden_nb_steps, truncated_bptt_ratio,
                                   spike_fn, neuron_model, neuron_parameters, weight_init,
                                   upsample_flag, upsample_mode, scale_flag, scale_factor, bn_flag, dropout_flag_, dropout_p,
                                   device, dtype, layer_index))

        # --------------------------------------------- Bottleneck block -----------------------------------------------
            # -------------------------------------------- Residual block ----------------------------------------------
        hidden_input_dim = hidden_output_dim
        hidden_output_dim, hidden_nb_steps = compute_conv_out_shape(hidden_input_dim, hidden_nb_steps,
                                                                    residual_kernel_size, residual_stride,
                                                                    residual_padding, residual_dilation)
        # -------------------------------------------------------------
        for i in range(nb_residual_block):
            # --------------------------
            layer_index += 1
            # --------------------------
            layers_list.append(
                SpikingResConv2dBlock(hidden_input_dim, hidden_output_dim, hidden_channels_list[-1],
                                      hidden_channels_list[-1], residual_kernel_size, residual_stride,
                                      residual_padding, residual_dilation, bias, padding_mode,
                                      hidden_nb_steps, truncated_bptt_ratio, spike_fn,
                                      neuron_model, neuron_parameters, weight_init,
                                      scale_flag, scale_factor, bn_flag, dropout_flag_, dropout_p,
                                      residual_skip_connection_type, device, dtype, layer_index))

        # ----------------------------------------------- Decoder block ------------------------------------------------
        hidden_channels_list.reverse()
        kernel_size_list.reverse()
        if pooling_flag:
            stride_list = pooling_stride_list
        stride_list.reverse()
        stride_list_ = dilation_list
        padding_list.reverse()
        hidden_dim.reverse()
        # --------------------------
        pooling_flag = False
        pooling_stride = None
        pooling_padding = None
        # --------------------------
        for i in range(nb_encoder_layers):
            # --------------------------
            layer_index += 1
            # --------------------------
            hidden_input_dim = hidden_output_dim
            # --------------------------
            upsample_flag_ = upsample_flag
            if stride_list[i][0] != 1:
                upsample_flag_ = True
                hidden_input_dim *= stride_list[i][0]
            if stride_list[i][1] != 1:
                upsample_flag_ = True
            # --------------------------
            hidden_output_dim, hidden_nb_steps = hidden_dim[i + 1]
            # --------------------------
            if i == 0:
                cat_scale_factor = 1
            elif skip_connection_type == 'cat_':
                cat_scale_factor = 2
            # --------------------------
            if i > 2:
                dropout_flag = False
            # --------------------------
            layers_list.append(
                SpikingConv2dBlock(hidden_input_dim, hidden_output_dim, cat_scale_factor * hidden_channels_list[i],
                                   hidden_channels_list[i + 1], kernel_size_list[i], stride_list_[i], padding_list[i],
                                   dilation_list[i], bias, padding_mode,
                                   pooling_flag, pooling_type, pooling_stride, pooling_padding,
                                   hidden_nb_steps, truncated_bptt_ratio, spike_fn,
                                   neuron_model, neuron_parameters, weight_init,
                                   upsample_flag_, upsample_mode, scale_flag, scale_factor, bn_flag, dropout_flag, dropout_p,
                                   device, dtype, layer_index))

        # ---------------------------------------- Intermediate Readout Layer ------------------------------------------
            if i < (nb_encoder_layers - 1) and use_intermediate_output:
                upsample_flag = True
                layer_index += 1
                # --------------------------
                if skip_connection_type == 'cat_':
                    cat_scale_factor = 2
                # --------------------------
                layers_list.append(
                    ReadoutConv2dBlock(output_dim, output_dim, cat_scale_factor * hidden_channels_list[i+1],
                                       output_channels, kernel_size_list[-1], stride_list_[-1], padding_list[-1],
                                       dilation_list[-1], bias, padding_mode,
                                       nb_steps, truncated_bptt_ratio, neuron_model, neuron_parameters,
                                       weight_init, upsample_flag, upsample_mode,
                                       scale_flag, scale_factor, device, dtype, layer_index))


        # ----------------------------------------------- Readout Layer ------------------------------------------------
        upsample_flag = True
        layer_index += 1
        # --------------------------
        if skip_connection_type == 'cat_':
            cat_scale_factor = 2
        # --------------------------
        layers_list.append(
            ReadoutConv2dBlock(output_dim, output_dim, cat_scale_factor * hidden_channels_list[-1],
                               output_channels, kernel_size_list[-1], stride_list_[-1], padding_list[-1],
                               dilation_list[-1], bias, padding_mode,
                               nb_steps, truncated_bptt_ratio, neuron_model, neuron_parameters,
                               weight_init, upsample_flag, upsample_mode,
                               scale_flag, scale_factor, device, dtype, layer_index))

        # --------------------------
        hidden_channels_list.reverse()
        # --------------------------

        super(ResBottleneckUNetSNN, self).__init__(layers_list, neuron_parameters['train_neuron_parameters'])

        self.skip_connection_type = skip_connection_type
        self.nb_skip_connection = nb_encoder_layers
        self.nb_residual_skip_connection = nb_residual_block
        self.use_intermediate_output = use_intermediate_output

    def forward(self, spk: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Method that defines the performed computation during the forward pass.
        """

        self.spk_rec.append(spk)
        output = []

        skip_connection_k = 0
        for i, snn_layer in enumerate(self.snn_layers):

            if isinstance(snn_layer, SpikingConv2dBlock) or isinstance(snn_layer, SpikingResConv2dBlock):
                if i <= (self.nb_skip_connection + self.nb_residual_skip_connection + 1):
                    spk, mem = snn_layer(spk)
                else:
                    if not self.use_intermediate_output:
                        spk_rec_k_index = self.nb_skip_connection - skip_connection_k
                        self.spk_skip_connection(spk, spk_rec_k_index)
                        skip_connection_k += 1

                    spk, mem = snn_layer(self.spk_rec[-1])

                if self.save_mem:
                    self.mem_rec.append(mem.clone().detach().cpu().numpy())

                self.spk_rec.append(spk)

            elif isinstance(snn_layer, ReadoutConv2dBlock):

                if self.use_intermediate_output:
                    spk_rec_k_index = self.nb_skip_connection - skip_connection_k
                    self.spk_skip_connection(spk, spk_rec_k_index)
                    skip_connection_k += 1

                    mem = snn_layer(self.spk_rec[-1])

                    output.append(mem)

                else:
                    self.spk_skip_connection(spk, 1)
                    mem = snn_layer(self.spk_rec[-1])

                    if self.save_mem:
                        self.mem_rec.append(mem.clone().detach().cpu().numpy())

        self._detach_rec()

        return mem, output

    def spk_skip_connection(self, spk: torch.Tensor, spk_rec_k_index: int) -> None:
        """Method that creates spiking skip connections output.

        Parameters
        ----------
        spk: torch.Tensor
            Input spike tensor of corresponding layer.
        spk_rec_k_index: int
            Spike tensor index within spike records.
        """
        if self.skip_connection_type == 'cat_':
            self.spk_rec[-1] = torch.cat((spk, self.spk_rec[spk_rec_k_index]), 1)
        elif self.skip_connection_type == 'add_':
            self.spk_rec[-1] = torch.add(spk, self.spk_rec[spk_rec_k_index])

