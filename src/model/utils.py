import numpy as np
import torch
from src.model.constants import ResidualConv2dParameters


def compute_conv_out_shape(input_dim, nb_steps, kernel_size, stride, padding, dilation):

    # Compute hidden layer dimensions
    output_dim = (((input_dim + (2 * padding[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1) / stride[0]) + 1)
    nb_steps = (((nb_steps + (2 * padding[1]) - (dilation[1] * (kernel_size[1] - 1)) - 1) / stride[1]) + 1)

    return int(np.floor(output_dim)), int(np.floor(nb_steps))


def set_conv2d_parameters(Conv2dParameters, hidden_channels_list, kernel_size: tuple, stride: tuple, padding: tuple,
                          dilation: tuple, use_same_layer: bool, pooling_flag: bool = False):
    if hidden_channels_list is None:
        hidden_channels_list = Conv2dParameters.hidden_channels_list
        kernel_size_list = Conv2dParameters.kernel_size_list
        if not pooling_flag:
            stride_list = Conv2dParameters.stride_list
        else:
            stride_list = [(1, 1) for i in range(len(hidden_channels_list))]
        dilation_list = Conv2dParameters.dilation_list
        padding_list = []
        for i in range(len(hidden_channels_list)):
            padding__0_i = int(
                np.ceil(((dilation_list[i][0] * (kernel_size_list[i][0] - 1)) + 1 - stride_list[i][0]) / 2))
            padding__1_i = int(
                np.ceil(((dilation_list[i][1] * (kernel_size_list[i][1] - 1)) + 1 - stride_list[i][1]) / 2))
            padding_list.append((padding__0_i, padding__1_i))
    else:
        n = len(hidden_channels_list)
        kernel_size_list = [kernel_size for i in range(n)]
        dilation_list = [dilation for i in range(n)]
        if not pooling_flag:
            stride_list = [stride for i in range(n)]
            padding_list = [padding for i in range(n)]
        else:
            stride_list = [(1, 1) for i in range(n)]
            padding_list = []
            for i in range(len(hidden_channels_list)):
                padding__0_i = int(
                    np.ceil(((dilation_list[i][0] * (kernel_size_list[i][0] - 1)) + 1 - stride_list[i][0]) / 2))
                padding__1_i = int(
                    np.ceil(((dilation_list[i][1] * (kernel_size_list[i][1] - 1)) + 1 - stride_list[i][1]) / 2))
                padding_list.append((padding__0_i, padding__1_i))

    if use_same_layer:
        hidden_channels_list.insert(0, int(hidden_channels_list[0] / 2))
        kernel_size_list.insert(0, kernel_size_list[0])
        stride_list.insert(0, (1, 1))
        dilation_list.insert(0, dilation_list[0])
        padding__0_0 = int(
            np.ceil(((dilation_list[0][0] * (kernel_size_list[0][0] - 1)) + 1 - stride_list[0][0]) / 2))
        padding__1_0 = int(
            np.ceil(((dilation_list[0][1] * (kernel_size_list[0][1] - 1)) + 1 - stride_list[0][1]) / 2))
        padding_list.insert(0, (padding__0_0, padding__1_0))

    return hidden_channels_list, kernel_size_list, stride_list, padding_list, dilation_list

def set_pooling2d_parameters(Conv2dParameters, hidden_channels_list, kernel_size: tuple, stride, use_same_layer: bool):
    if hidden_channels_list is None:
        hidden_channels_list = Conv2dParameters.hidden_channels_list
        kernel_size_list = Conv2dParameters.kernel_size_list
        pooling_stride_list = Conv2dParameters.stride_list
    else:
        n = len(hidden_channels_list)
        kernel_size_list = [kernel_size for i in range(n)]
        pooling_stride_list = [stride for i in range(n)]
    pooling_dilation = (1, 1)
    pooling_padding_list = []
    for i in range(len(hidden_channels_list)):
        padding__0_i = int(
            np.ceil(((pooling_dilation[0] * (kernel_size_list[i][0] - 1)) + 1 - pooling_stride_list[i][0]) / 2))
        padding__1_i = int(
            np.ceil(((pooling_dilation[1] * (kernel_size_list[i][1] - 1)) + 1 - pooling_stride_list[i][1]) / 2))
        pooling_padding_list.append((padding__0_i, padding__1_i))

    if use_same_layer:
        pooling_stride_list.insert(0, (1, 1))
        padding__0_0 = int(
            np.ceil(((pooling_dilation[0] * (kernel_size_list[0][0] - 1)) + 1 - pooling_stride_list[0][0]) / 2))
        padding__1_0 = int(
            np.ceil(((pooling_dilation[1] * (kernel_size_list[0][1] - 1)) + 1 - pooling_stride_list[0][1]) / 2))
        pooling_padding_list.insert(0, (padding__0_0, padding__1_0))
    return pooling_stride_list, pooling_padding_list


def set_residual_conv2d_parameters():
    kernel_size = ResidualConv2dParameters.residual_kernel_size
    stride = ResidualConv2dParameters.residual_stride
    dilation = ResidualConv2dParameters.residual_dilation
    padding_0 = int(np.ceil(((dilation[0] * (kernel_size[0] - 1)) + 1 - stride[0]) / 2))
    padding_1 = int(np.ceil(((dilation[1] * (kernel_size[1] - 1)) + 1 - stride[1]) / 2))
    padding = (padding_0, padding_1)
    return kernel_size, stride, padding, dilation


def pad_coefficients(coefficients: torch.Tensor):
    # coefficients_ = F.pad(input=coefficients, pad=(0, 0, 0, 1), mode='constant', value=0)
    coefficients_ = torch.cat((coefficients, coefficients[:, :, -1:, :]), 2)
    return coefficients_


def complex_spectrogram(magnitude: torch.Tensor, phase: torch.Tensor):
    complex_spectrogram_list = []

    for i in range(magnitude.shape[0]):
        complex_spectrogram_list.append(magnitude[i] * torch.exp(1j * phase[i]))

    return torch.stack(complex_spectrogram_list, 0)


def random_index(n: int, high: int, low: int = 0):
    # assert (high - low) >= n, f'high (={high}) - low (={low}) value must be greater than n (={n})'
    n = min(n, high-low)

    unique_set = set()
    random_integers = []

    while len(unique_set) < n:
        random_int = np.random.randint(low, high)
        if random_int not in unique_set:
            unique_set.add(random_int)
            random_integers.append(random_int)

    return np.array(random_integers)

