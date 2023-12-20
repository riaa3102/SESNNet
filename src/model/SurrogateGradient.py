"""
File:
    model/SurrogateGradient.py

Description:
    Defines the SuperSpike class, SigmoidDerivative class, ATan class and PiecewiseLinear class.
"""

import torch
import math
import torch.nn.functional as F


def heaviside(input_: torch.Tensor, spiking_mode: str) -> torch.Tensor:
    """Method that computes the Heaviside step function.

    Parameters
    ----------
    input_: torch.Tensor
        Input tensor.
    spiking_mode: str
        'binary' or 'graded'.
    """
    out = torch.zeros_like(input_)
    if spiking_mode == 'binary':
        # Heaviside step function
        out[input_ >= 0] = 1.
    elif spiking_mode == 'graded':
        out = F.relu(input_)
    return out


class SuperSpike(torch.autograd.Function):
    """Class that implements the SuperSpike function as the surrogate gradient for the backward pass.
    """

    # scale parameter
    surrogate_scale: float = 10.  # controls steepness of surrogate gradient

    # Spiking mode parameter
    spiking_mode: str = 'binary'

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using 'Heaviside step function'
        """
        # ctx: use to stash information for backward computation
        ctx.save_for_backward(input_)
        return heaviside(input_, SuperSpike.spiking_mode)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass using surrogate gradient function: SuperSpike
        """
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Normalized negative part of a fast sigmoid: Zenke & Ganguli (2018)
        grad = grad_input / ((1. + SuperSpike.surrogate_scale * torch.abs(input_)) ** 2)
        return grad


class SigmoidDerivative(torch.autograd.Function):
    """Class that implements the sigmoid derivative function as the surrogate gradient for the backward pass.
    """

    # scale parameter
    surrogate_scale: float = 2.  # controls steepness of surrogate gradient

    # Spiking mode parameter
    spiking_mode: str = 'binary'

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using 'Heaviside step function'
        """
        # ctx: use to stash information for backward computation
        ctx.save_for_backward(input_)
        return heaviside(input_, SigmoidDerivative.spiking_mode)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass using surrogate gradient function: sigmoid derivative
        """
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Modified sigmoid derivative
        grad = grad_input * SigmoidDerivative.surrogate_scale * torch.exp(-SigmoidDerivative.surrogate_scale * input_) / (
                (1. + torch.exp(-SigmoidDerivative.surrogate_scale * input_)) ** 2)
        return grad


class ATan(torch.autograd.Function):
    """Class that implements the SuperSpike function as the surrogate gradient for the backward pass.
    """

    # scale parameter
    surrogate_scale = 2.

    # Spiking mode parameter
    spiking_mode = 'binary'

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using 'Heaviside step function'
        """
        # ctx: use to stash information for backward computation
        ctx.save_for_backward(input_)
        return heaviside(input_, ATan.spiking_mode)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass using surrogate gradient function:
        """
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * (1. / math.pi) * (1. / (1. + ((math.pi * input_ * (ATan.surrogate_scale / 2.)) ** 2)))
        return grad


class PiecewiseLinear(torch.autograd.Function):
    """Class that implements the piecewise linear function (Esser et al.) as the surrogate gradient for the backward
    pass.
    """

    # scale parameter
    surrogate_scale: float = 1.  # controls steepness of surrogate gradient

    # Spiking mode parameter
    spiking_mode: str = 'binary'

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using 'Heaviside step function'
        """
        # ctx: use to stash information for backward computation
        ctx.save_for_backward(input_)
        return heaviside(input_, PiecewiseLinear.spiking_mode)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass using surrogate gradient function: piecewise linear function (Esser et al.)
        """
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Piecewise linear function (Esser et al.)
        grad_input[input_ < 0] = 0.
        grad = grad_input * torch.max(torch.tensor([0], dtype=input_.dtype, device=input_.device), (1. - (PiecewiseLinear.surrogate_scale * torch.abs(input_))))
        return grad


class SpikeFunc(torch.autograd.Function):
    """Class that implements the SpikeFunc function as the surrogate gradient for the backward pass.
    """

    # scale parameter
    surrogate_scale: float = 10.  # controls steepness of surrogate gradient

    # Spiking mode parameter
    spiking_mode: str = 'binary'

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using 'Heaviside step function'
        """
        # ctx: use to stash information for backward computation
        ctx.save_for_backward(input_)
        return heaviside(input_, SpikeFunc.spiking_mode)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass using surrogate gradient function: SuperSpike
        """
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_ < 0] *= F.relu(1 + input_[input_ < 0])
        return grad_input
