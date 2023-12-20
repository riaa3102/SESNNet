"""
File:
    model/constants.py

Description:
    This file stores helpful constants value.
"""

from typing import List


class SNNstr(str):
    FCSNNstr = 'FCSNN'
    CSNNstr = 'CSNN'
    UNetSNNstr = 'UNetSNN'
    ResBottleneckUNetSNNstr = 'ResBottleneckUNetSNN'


class ANNstr(str):
    CNNstr = 'CNN'
    UNetstr = 'UNet'
    ResBottleneckUNetstr = 'ResBottleneckUNet'


class Linear1dParameters(List):
    hidden_dim_list = [512, 512]


class Conv2dParameters(List):
    hidden_channels_list = [64, 128, 256, 512, 512, 512, 512, 512]
    kernel_size_list = [(7, 5), (7, 5), (7, 5), (5, 5), (5, 5), (3, 3), (3, 3), (3, 3)]
    stride_list = [(2, 1), (2, 1), (2, 1), (2, 1), (2, 2), (2, 2), (2, 2), (2, 2)]
    dilation_list = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]


class UnetConv2dParameters(List):
    hidden_channels_list = [64, 128, 256, 512, 512, 512, 512, 512]
    kernel_size_list = [(7, 5), (7, 5), (7, 5), (5, 5), (5, 5), (3, 3), (3, 3), (3, 3)]
    stride_list = [(2, 1), (2, 1), (2, 1), (2, 1), (2, 2), (2, 2), (2, 2), (2, 2)]
    dilation_list = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]


class ResidualConv2dParameters(tuple):
    residual_kernel_size = (3, 3)
    residual_stride = (1, 1)
    residual_dilation = (1, 1)


class ActivationParameters(float):
    negative_slope = 0.2

