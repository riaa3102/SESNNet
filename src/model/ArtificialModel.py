"""
File:
    model/ArtificialModel.py

Description:
    Defines the FCSNN, ConvSNN class.
"""

from typing import Optional, Union, List, Tuple
import numpy as np
import math
import torch
import torch.nn as nn
from src.model.constants import Conv2dParameters, UnetConv2dParameters
from src.model.utils import set_conv2d_parameters, set_residual_conv2d_parameters, compute_conv_out_shape
from src.model.ArtificialBlock import FCBlock, Conv2dBlock, ResConv2dBlock


class ANNBase(nn.Module):
    """Class that implements the SNN model architecture as a set of sequential ANN layers.
    """

    def __init__(self, layers_list: nn.ModuleList):
        """Method that defines the performed computation during the forward pass.

        Parameters
        ----------
        layers_list: nn.ModuleList
            A list of spiking layers that is used to build the ANN model.
        """

        super(ANNBase, self).__init__()

        self.ann_layers = layers_list
        self.nb_layers = len(layers_list)
        self.layers_name = self._layers_name()
        self.out_rec = []

    def init_rec(self):
        self.out_rec = []

    def _detach_rec(self):
        for i in range(len(self.out_rec)):
                self.out_rec[i] = self.out_rec[i].clone().detach().cpu().numpy()

    def _layers_name(self):
        ann_layer_name_list = []
        for ann_layer_i in self.ann_layers:
            ann_layer_name = ann_layer_i.__class__.__name__
            for ann_layer_i_j in ann_layer_i.children():
                if isinstance(ann_layer_i_j, nn.Sequential):
                    ann_layer_name_list.append(ann_layer_name)
        return ann_layer_name_list


class CNN(ANNBase):
    """Class that implements the SNN model architecture with predefined list of convolutional layers.
    """

    def __init__(self, input_dim: int, hidden_channels_list: list, output_dim: int, kernel_size: tuple, stride: tuple,
                 padding: tuple, dilation: tuple, bias: bool, use_same_layer: bool, nb_steps: int,
                 activation_fn: list, weight_init: dict, upsample_mode: str = 'bilinear',
                 scale_flag: bool = True, scale_factor: float = 1.,
                 bn_flag: bool = False, dropout_flag: bool = False, dropout_p: float = 0.5,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float):

        # ------------------------------------------------ Init Params -------------------------------------------------

        hidden_channels_list, kernel_size_list, stride_list, padding_list, dilation_list= set_conv2d_parameters(
            UnetConv2dParameters, hidden_channels_list, kernel_size, stride, padding, dilation, use_same_layer)

        hidden_dim = []

        # ------------------------------------------------

        nb_encoder_layers = len(hidden_channels_list) - 1

        layers_list = nn.ModuleList()
        input_channels = 1
        output_channels = 1

        # ------------------------------------------------ Input Layer -------------------------------------------------
        layer_index = 0
        upsample_flag = False
        dropout_flag_ = False
        # --------------------------
        # Compute hidden layer output
        hidden_output_dim, hidden_nb_steps = compute_conv_out_shape(input_dim, nb_steps, kernel_size_list[0],
                                                                    stride_list[0], padding_list[0], dilation_list[0])
        hidden_dim.append((hidden_output_dim, hidden_nb_steps))
        # --------------------------
        layers_list.append(
            Conv2dBlock(input_dim, hidden_output_dim, input_channels, hidden_channels_list[0], kernel_size_list[0],
                        stride_list[0], padding_list[0], dilation_list[0], bias, hidden_nb_steps,
                        activation_fn[0], weight_init, upsample_flag, upsample_mode, scale_flag, scale_factor,
                        bn_flag, dropout_flag_, dropout_p, device, dtype, layer_index))

        # ---------------------------------------------- Hidden Layer(s) -----------------------------------------------
        for i in range(nb_encoder_layers):
            # --------------------------
            layer_index += 1
            # --------------------------
            hidden_input_dim = hidden_output_dim
            hidden_output_dim, hidden_nb_steps = compute_conv_out_shape(hidden_input_dim, hidden_nb_steps,
                                                                        kernel_size_list[i + 1], stride_list[i + 1],
                                                                        padding_list[i + 1], dilation_list[i + 1])
            hidden_dim.append((hidden_output_dim, hidden_nb_steps))
            # --------------------------
            layers_list.append(
                Conv2dBlock(hidden_input_dim, hidden_output_dim, hidden_channels_list[i],
                            hidden_channels_list[i + 1], kernel_size_list[i+1], stride_list[i+1], padding_list[i+1],
                            dilation_list[i+1], bias, hidden_nb_steps, activation_fn[0], weight_init,
                            upsample_flag, upsample_mode, scale_flag, scale_factor, bn_flag, dropout_flag_, dropout_p,
                            device, dtype, layer_index))

        # ---------------------------------------------- Hidden Layer(s) -----------------------------------------------
        hidden_channels_list.reverse()
        kernel_size_list.reverse()
        stride_list.reverse()
        stride_list_ = dilation_list
        padding_list.reverse()
        hidden_dim.reverse()
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
                Conv2dBlock(hidden_input_dim, hidden_output_dim, hidden_channels_list[i],
                            hidden_channels_list[i + 1], kernel_size_list[i], stride_list_[i], padding_list[i],
                            dilation_list[i], bias, hidden_nb_steps, activation_fn[0],
                            weight_init, upsample_flag_, upsample_mode, scale_flag, scale_factor, bn_flag,
                            dropout_flag, dropout_p, device, dtype, layer_index))

        # ----------------------------------------------- Readout Layer ------------------------------------------------
        bn_flag = False
        dropout_flag = False
        upsample_flag = True
        layer_index += 1
        # --------------------------
        layers_list.append(
            Conv2dBlock(output_dim, output_dim, hidden_channels_list[-1],
                        output_channels, kernel_size_list[-1], stride_list_[-1], padding_list[-1], dilation_list[-1],
                        bias, nb_steps, activation_fn[-1], weight_init, upsample_flag, upsample_mode, scale_flag,
                        scale_factor, bn_flag, dropout_flag, dropout_p, device, dtype, layer_index))

        # --------------------------
        hidden_channels_list.reverse()
        # --------------------------

        super(CNN, self).__init__(layers_list)

    def forward(self, x: torch.Tensor):
        """Method that defines the performed computation during the forward pass.
        """

        self.out_rec.append(x)

        for i, ann_layer in enumerate(self.ann_layers):
            x = ann_layer(x)
            self.out_rec.append(x)

        self._detach_rec()

        return x, []


class UNet(ANNBase):
    """Class that implements the SNN model architecture with predefined list of convolutional layers.
    """

    def __init__(self, input_dim: int, hidden_channels_list: list, output_dim: int, kernel_size: tuple, stride: tuple,
                 padding: tuple, dilation: tuple, bias: bool, use_same_layer: bool, nb_steps: int,
                 activation_fn: list, weight_init: dict, upsample_mode: str = 'bilinear',
                 scale_flag: bool = True, scale_factor: float = 1.,
                 bn_flag: bool = False, dropout_flag: bool = False, dropout_p: float = 0.5,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float,
                 skip_connection_type: str = 'cat_'):

        # ------------------------------------------------ Init Params -------------------------------------------------

        hidden_channels_list, kernel_size_list, stride_list, padding_list, dilation_list= set_conv2d_parameters(
            UnetConv2dParameters, hidden_channels_list, kernel_size, stride, padding, dilation, use_same_layer)

        hidden_dim = []

        # ------------------------------------------------

        nb_encoder_layers = len(hidden_channels_list) - 1

        layers_list = nn.ModuleList()
        input_channels = 1
        output_channels = 1

        # ------------------------------------------------ Input Layer -------------------------------------------------
        layer_index = 0
        upsample_flag = False
        dropout_flag_ = False
        # --------------------------
        # Compute hidden layer output
        hidden_output_dim, hidden_nb_steps = compute_conv_out_shape(input_dim, nb_steps, kernel_size_list[0],
                                                                    stride_list[0], padding_list[0], dilation_list[0])
        hidden_dim.append((hidden_output_dim, hidden_nb_steps))
        # --------------------------
        layers_list.append(
            Conv2dBlock(input_dim, hidden_output_dim, input_channels, hidden_channels_list[0], kernel_size_list[0],
                        stride_list[0], padding_list[0], dilation_list[0], bias, hidden_nb_steps,
                        activation_fn[0], weight_init, upsample_flag, upsample_mode, scale_flag, scale_factor,
                        bn_flag, dropout_flag_, dropout_p, device, dtype, layer_index))

        # ----------------------------------------------- Encoder block ------------------------------------------------
        for i in range(nb_encoder_layers):
            # --------------------------
            layer_index += 1
            # --------------------------
            hidden_input_dim = hidden_output_dim
            hidden_output_dim, hidden_nb_steps = compute_conv_out_shape(hidden_input_dim, hidden_nb_steps,
                                                                        kernel_size_list[i + 1], stride_list[i + 1],
                                                                        padding_list[i + 1], dilation_list[i + 1])
            hidden_dim.append((hidden_output_dim, hidden_nb_steps))
            # --------------------------
            layers_list.append(
                Conv2dBlock(hidden_input_dim, hidden_output_dim, hidden_channels_list[i],
                            hidden_channels_list[i + 1], kernel_size_list[i+1], stride_list[i+1], padding_list[i+1],
                            dilation_list[i+1], bias, hidden_nb_steps, activation_fn[0], weight_init,
                            upsample_flag, upsample_mode, scale_flag, scale_factor, bn_flag, dropout_flag_, dropout_p,
                            device, dtype, layer_index))

        # ----------------------------------------------- Decoder block ------------------------------------------------
        hidden_channels_list.reverse()
        kernel_size_list.reverse()
        stride_list.reverse()
        stride_list_ = dilation_list
        padding_list.reverse()
        hidden_dim.reverse()
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
                Conv2dBlock(hidden_input_dim, hidden_output_dim, cat_scale_factor * hidden_channels_list[i],
                            hidden_channels_list[i + 1], kernel_size_list[i], stride_list_[i], padding_list[i],
                            dilation_list[i], bias, hidden_nb_steps, activation_fn[0],
                            weight_init, upsample_flag_, upsample_mode, scale_flag, scale_factor, bn_flag,
                            dropout_flag, dropout_p, device, dtype, layer_index))

        # ----------------------------------------------- Readout Layer ------------------------------------------------
        bn_flag = False
        dropout_flag = False
        upsample_flag = True
        layer_index += 1
        # --------------------------
        if skip_connection_type == 'cat_':
            cat_scale_factor = 2
        # --------------------------
        layers_list.append(
            Conv2dBlock(output_dim, output_dim, cat_scale_factor * hidden_channels_list[-1],
                        output_channels, kernel_size_list[-1], stride_list_[-1], padding_list[-1], dilation_list[-1],
                        bias, nb_steps, activation_fn[-1], weight_init, upsample_flag, upsample_mode, scale_flag,
                        scale_factor, bn_flag, dropout_flag, dropout_p, device, dtype, layer_index))

        # --------------------------
        hidden_channels_list.reverse()
        # --------------------------

        super(UNet, self).__init__(layers_list)

        self.skip_connection_type = skip_connection_type
        self.nb_skip_connection = nb_encoder_layers

    def forward(self, x: torch.Tensor):
        """Method that defines the performed computation during the forward pass.
        """

        self.out_rec.append(x)

        skip_connection_k = 0
        for i, ann_layer in enumerate(self.ann_layers):

            if i <= (self.nb_skip_connection + 1):
                x = ann_layer(x)
            else:
                x_rec_k_index = self.nb_skip_connection - skip_connection_k
                x = self.x_skip_connection(x, x_rec_k_index)
                self.out_rec[-1] = x
                skip_connection_k += 1

                x = ann_layer(x)

            self.out_rec.append(x)

        self._detach_rec()

        return x, []

    def x_skip_connection(self, x, x_rec_k_index: int):

        if self.skip_connection_type == 'cat_':
            x = torch.cat((x, self.out_rec[x_rec_k_index]), 1)
        elif self.skip_connection_type == 'add_':
            x = torch.add(x, self.out_rec[x_rec_k_index])
        return x


class ResBottleneckUNet(ANNBase):
    """Class that implements the SNN model architecture with predefined list of convolutional layers.
    """

    def __init__(self, input_dim: int, hidden_channels_list: list, output_dim: int, kernel_size: tuple, stride: tuple,
                 padding: tuple, dilation: tuple, bias: bool, use_same_layer: bool, nb_steps: int,
                 activation_fn: list, weight_init: dict, upsample_mode: str = 'bilinear', scale_flag: bool = True,
                 scale_factor: float = 1., bn_flag: bool = False, dropout_flag: bool = False, dropout_p: float = 0.5,
                 device: torch.device = 'cpu', dtype: torch.dtype = torch.float,
                 skip_connection_type: str = 'cat_', nb_residual_block: int = 1,
                 residual_skip_connection_type: str = 'add_'):

        # ------------------------------------------------ Init Params -------------------------------------------------

        hidden_channels_list, kernel_size_list, stride_list, padding_list, dilation_list = set_conv2d_parameters(
            UnetConv2dParameters, hidden_channels_list, kernel_size, stride, padding, dilation, use_same_layer)
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
        hidden_output_dim, hidden_nb_steps = compute_conv_out_shape(input_dim, nb_steps,
                                                                    kernel_size_list[0], stride_list[0],
                                                                    padding_list[0], dilation_list[0])
        hidden_dim.append((hidden_output_dim, hidden_nb_steps))
        # --------------------------
        layers_list.append(
            Conv2dBlock(input_dim, hidden_output_dim, input_channels, hidden_channels_list[0], kernel_size_list[0],
                        stride_list[0], padding_list[0], dilation_list[0], bias, hidden_nb_steps, activation_fn[0],
                        weight_init, upsample_flag, upsample_mode, scale_flag, scale_factor, bn_flag, dropout_flag_,
                        dropout_p, device, dtype, layer_index))

        # ----------------------------------------------- Encoder block ------------------------------------------------
        for i in range(nb_encoder_layers):
            # --------------------------
            layer_index += 1
            # --------------------------
            hidden_input_dim = hidden_output_dim
            hidden_output_dim, hidden_nb_steps = compute_conv_out_shape(hidden_input_dim, hidden_nb_steps,
                                                                        kernel_size_list[i + 1], stride_list[i + 1],
                                                                        padding_list[i + 1], dilation_list[i + 1])
            hidden_dim.append((hidden_output_dim, hidden_nb_steps))
            # -------------------------------------------------------------
            layers_list.append(
                Conv2dBlock(hidden_input_dim, hidden_output_dim, hidden_channels_list[i],
                            hidden_channels_list[i + 1], kernel_size_list[i + 1], stride_list[i + 1],
                            padding_list[i + 1], dilation_list[i + 1], bias, hidden_nb_steps,
                            activation_fn[0], weight_init, upsample_flag, upsample_mode, scale_flag, scale_factor,
                            bn_flag, dropout_flag_, dropout_p, device, dtype, layer_index))

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
                ResConv2dBlock(hidden_input_dim, hidden_output_dim, hidden_channels_list[-1],
                               hidden_channels_list[-1], residual_kernel_size, residual_stride,
                               residual_padding, residual_dilation, bias, activation_fn[0], weight_init,
                               scale_flag, scale_factor, bn_flag, dropout_flag_, dropout_p,
                               residual_skip_connection_type, device, dtype, layer_index))

        # ----------------------------------------------- Decoder block ------------------------------------------------
        hidden_channels_list.reverse()
        kernel_size_list.reverse()
        stride_list.reverse()
        stride_list_ = dilation_list
        padding_list.reverse()
        hidden_dim.reverse()
        # -------------------------------------------------------------
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
                Conv2dBlock(hidden_input_dim, hidden_output_dim, cat_scale_factor * hidden_channels_list[i],
                            hidden_channels_list[i + 1], kernel_size_list[i], stride_list_[i], padding_list[i],
                            dilation_list[i], bias, hidden_nb_steps, activation_fn[0], weight_init, upsample_flag_,
                            upsample_mode, scale_flag, scale_factor, bn_flag, dropout_flag, dropout_p,
                            device, dtype, layer_index))

        # ----------------------------------------------- Readout Layer ------------------------------------------------
        bn_flag = False
        dropout_flag = False
        upsample_flag = True
        layer_index += 1
        # --------------------------
        if skip_connection_type == 'cat_':
            cat_scale_factor = 2
        # --------------------------
        layers_list.append(
            Conv2dBlock(output_dim, output_dim, cat_scale_factor * hidden_channels_list[-1],
                        output_channels, kernel_size_list[-1], stride_list_[-1], padding_list[-1], dilation_list[-1],
                        bias, nb_steps, activation_fn[-1], weight_init, upsample_flag,
                        upsample_mode, scale_flag, scale_factor, bn_flag, dropout_flag, dropout_p,
                        device, dtype, layer_index))

        # --------------------------
        hidden_channels_list.reverse()
        # --------------------------

        super(ResBottleneckUNet, self).__init__(layers_list)

        self.skip_connection_type = skip_connection_type
        self.nb_skip_connection = nb_encoder_layers
        self.nb_residual_skip_connection = nb_residual_block

    def forward(self, x: torch.Tensor):
        """Method that defines the performed computation during the forward pass.
        """

        self.out_rec.append(x)

        skip_connection_k = 0
        for i, ann_layer in enumerate(self.ann_layers):

            if i <= (self.nb_skip_connection + self.nb_residual_skip_connection + 1):
                x = ann_layer(x)
            else:
                x_rec_k_index = self.nb_skip_connection - skip_connection_k
                self.x_skip_connection(x, x_rec_k_index)
                skip_connection_k += 1

                x = ann_layer(self.out_rec[-1])

            self.out_rec.append(x)

        self._detach_rec()

        return x, []

    def x_skip_connection(self, x, x_rec_k_index: int):

        if self.skip_connection_type == 'cat_':
            self.out_rec[-1] = torch.cat((x, self.out_rec[x_rec_k_index]), 1)
        elif self.skip_connection_type == 'add_':
            self.out_rec[-1] = torch.add(x, self.out_rec[x_rec_k_index])

