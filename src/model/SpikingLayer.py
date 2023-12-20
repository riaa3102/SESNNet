"""
File:
    model/SpikingLayer.py

Description:
    Defines the Linear1d class, ScaleLayer class, Recurrent1d, Recurrent2d, Upsampling2d, NeuronModel, MPNeuronModel,
    LIF1d, LI1d, LIF2d, LI2d, PLIF2d, PLI2d, KLIF2d, KLI2d IF2d and I2d class.
"""

from typing import Tuple, Callable, Union, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear1d(nn.Module):
    """Class that implements linear connections layer.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(Linear1d, self).__init__()

        self.weight = nn.Parameter(torch.empty((input_dim, output_dim)), requires_grad=True)

    def forward(self, x):
        return torch.einsum('abc,bd->adc', x, self.weight)


class ScaleLayer(nn.Module):
    """Class that implements a multiplication layer.
    """

    def __init__(self, scale_factor: float = 1., scale_flag: bool = False) -> None:
        super(ScaleLayer, self).__init__()

        self.scale_flag = scale_flag

        if scale_flag:
            self.scale_factor = nn.Parameter(torch.tensor([scale_factor]), requires_grad=True)
        else:
            self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor):
        return torch.mul(x, self.scale_factor)


class Recurrent1d(nn.Module):
    """Class that implements 1D-recurrent connections layer.
    """

    def __init__(self, output_dim: int) -> None:
        super(Recurrent1d, self).__init__()

        self.v = nn.Parameter(torch.empty((output_dim, output_dim)), requires_grad=True)

    def forward(self, output_spikes: torch.Tensor):
        return torch.einsum('ab,bc->ac', output_spikes, self.v)


class Recurrent2d(nn.Module):
    """Class that implements 2D-recurrent connections layer.
    """

    def __init__(self, output_channels: int) -> None:
        super(Recurrent2d, self).__init__()

        self.v = nn.Parameter(torch.empty((output_channels, output_channels)), requires_grad=True)

    def forward(self, output_spikes):
        return torch.einsum('abc,bd->adc', output_spikes, self.v)


class Upsampling2d(nn.Module):
    """Class that implements 2D-upsampling layer.
    """

    def __init__(self, input_dim: int, nb_steps: int, mode: str = 'bilinear') -> None:
        super(Upsampling2d, self).__init__()

        self.input_dim = input_dim
        self.nb_steps = nb_steps
        self.mode = mode                                            # 'nearest', 'bilinear'

    def forward(self, x):
        return F.interpolate(x,
                             size=(self.input_dim, self.nb_steps),
                             mode=self.mode,
                             align_corners=False if self.mode == 'bilinear' else None,
                             )


class NeuronModel(nn.Module):
    """Class that implements spiking neurons layer.
    """

    def __init__(self, truncated_bptt_ratio: int, spike_fn: Callable, reset_mode: str, detach_reset: bool,
                 decay_input: bool, device, dtype) -> None:

        super(NeuronModel, self).__init__()

        self.device = device
        self.dtype = dtype

        self.neuron_model = None
        self.nb_steps = None
        self.nb_steps_truncated_bptt = None
        self.truncated_bptt_ratio = truncated_bptt_ratio

        self.spike_fn = spike_fn
        self.reset_mode = reset_mode
        self.detach_reset = detach_reset
        self.decay_input = decay_input

        self._save_mem = False

        self.states_dict = {'synaptic_current': None,
                            'membrane_potential': None,
                            'output_spikes': None}

    @property
    def save_mem(self):
        return self._save_mem

    @save_mem.setter
    def save_mem(self, value: bool):
        self._save_mem = value

    def forward(self, input_: torch.Tensor) ->  Tuple[torch.Tensor, torch.Tensor]:
        return self.multi_step_neuron(input_)

    def multi_step_neuron(self, input_: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, List]]:
        pass

    def reset_neuron(self, new_membrane_potential: torch.Tensor, reset: torch.Tensor):
        """Method that handles the spiking neurons membrane potential reset.

        Parameters
        ----------
        new_membrane_potential: torch.Tensor
            New membrane potential tensor.
        reset: torch.Tensor
            Reset tensor.
        """
        if self.reset_mode == 'hard_reset':
            # new_membrane_potential = new_membrane_potential * (1. - reset)
            new_membrane_potential = new_membrane_potential * torch.where(reset != 0., 0., 1.)
        elif self.reset_mode == 'soft_reset':
            new_membrane_potential = new_membrane_potential - self.membrane_threshold * reset
        return new_membrane_potential

    def init_neuron_parameters(self, neuron_parameters: dict, lif_std: float = 0.01, lif_bound: float = 0.01):
        """Method that handles the spiking neurons parameters' initialization.

        Parameters
        ----------
        neuron_parameters: dict
            Neuron model specifications' dictionary.
        lif_std: float
            Initialization parameter.
        lif_bound: float
            Initialization parameter.
        """
        if neuron_parameters['neuron_parameters_init_dist'] == 'constant_':
            nn.init.constant_(self.membrane_threshold, val=neuron_parameters['membrane_threshold'])
            if self.neuron_model in ['lif', 'plif', 'klif']:
                nn.init.constant_(self.alpha, val=neuron_parameters['alpha'])
                nn.init.constant_(self.beta, val=neuron_parameters['beta'])
        elif neuron_parameters['neuron_parameters_init_dist'] == 'normal_':
            nn.init.normal_(self.membrane_threshold, mean=neuron_parameters['membrane_threshold'], std=lif_std)
            if self.neuron_model in ['lif', 'plif', 'klif']:
                nn.init.normal_(self.alpha, mean=neuron_parameters['alpha'], std=lif_std)
                nn.init.normal_(self.beta, mean=neuron_parameters['beta'], std=lif_std)
        elif neuron_parameters['neuron_parameters_init_dist'] == 'uniform_':
            nn.init.uniform_(self.membrane_threshold,
                             a=neuron_parameters['membrane_threshold'] - lif_bound,
                             b=neuron_parameters['membrane_threshold'] + lif_bound)
            if self.neuron_model in ['lif', 'plif', 'klif']:
                nn.init.uniform_(self.alpha,
                                 a=neuron_parameters['alpha'] - lif_bound, b=neuron_parameters['alpha'] + lif_bound)
                nn.init.uniform_(self.beta,
                                 a=neuron_parameters['beta'] - lif_bound, b=neuron_parameters['beta'] + lif_bound)

    def clamp_neuron_parameters(self):
        """Method that handles the spiking neurons parameters tensors' clamp.
        """
        self.membrane_threshold.data.clamp_(min=0.)
        if self.neuron_model == 'lif':
            self.alpha.data.clamp_(min=0., max=1.)
            self.beta.data.clamp_(min=0., max=1.)


class MPNeuronModel(nn.Module):
    """Class that implements non-spiking neurons layer.
    """

    def __init__(self, truncated_bptt_ratio: int, decay_input: bool, device, dtype) -> None:
        super(MPNeuronModel, self).__init__()

        self.device = device
        self.dtype = dtype

        self.neuron_model = None
        self.nb_steps = None
        self.nb_steps_truncated_bptt = None
        self.truncated_bptt_ratio = truncated_bptt_ratio

        self.decay_input = decay_input

        self.states_dict = {'synaptic_current': None,
                            'membrane_potential': None}

    def forward(self, input_: torch.Tensor):
        return self.multi_step_LI(input_)

    def multi_step_LI(self, input_: torch.Tensor) -> Tuple[torch.Tensor]:
        pass

    def init_neuron_parameters(self, neuron_parameters: dict, lif_std: float = 0.01, lif_bound: float = 0.01):
        """Method that handles the non-spiking neurons parameters' initialization.

        Parameters
        ----------
        neuron_parameters: dict
            Neuron model specifications' dictionary.
        lif_std: float
            Initialization parameter.
        lif_bound: float
            Initialization parameter.
        """
        if neuron_parameters['neuron_parameters_init_dist'] == 'constant_':
            if self.neuron_model in ['lif', 'plif', 'klif']:
                nn.init.constant_(self.alpha, val=neuron_parameters['alpha_out'])
                nn.init.constant_(self.beta, val=neuron_parameters['beta_out'])
        elif neuron_parameters['neuron_parameters_init_dist'] == 'normal_':
            if self.neuron_model in ['lif', 'plif', 'klif']:
                nn.init.normal_(self.alpha, mean=neuron_parameters['alpha_out'], std=lif_std)
                nn.init.normal_(self.beta, mean=neuron_parameters['beta_out'], std=lif_std)
        elif neuron_parameters['neuron_parameters_init_dist'] == 'uniform_':
            if self.neuron_model in ['lif', 'plif', 'klif']:
                nn.init.uniform_(self.alpha, a=neuron_parameters['alpha_out'] - lif_bound,
                                 b=neuron_parameters['alpha_out'] + lif_bound)
                nn.init.uniform_(self.beta, a=neuron_parameters['beta_out'] - lif_bound,
                                 b=neuron_parameters['beta_out'] + lif_bound)

    def clamp_neuron_parameters(self):
        """Method that handles the non-spiking neurons parameters tensors' clamp.
        """
        if self.neuron_model == 'lif':
            self.alpha.data.clamp_(min=0., max=1.)
            self.beta.data.clamp_(min=0., max=1.)


class LIF1d(NeuronModel):
    """Class that implements 1D-LIF neurons layer.
    """

    def __init__(self, output_dim: int, truncated_bptt_ratio: int, membrane_threshold, alpha, beta,
                 train_neuron_parameters_flag: bool, spike_fn: Callable, reset_mode: str, detach_reset: bool,
                 recurrent_flag: bool, decay_input: bool, device, dtype) -> None:

        super(LIF1d, self).__init__(truncated_bptt_ratio, spike_fn, reset_mode, detach_reset, decay_input, device, dtype)

        self.neuron_model = 'lif'

        self.output_dim = output_dim

        if train_neuron_parameters_flag:
            # Membrane threshold
            self.membrane_threshold = nn.Parameter(torch.empty(output_dim), requires_grad=True)
            # Membrane leak coefficients
            self.alpha = nn.Parameter(torch.empty(output_dim), requires_grad=True)
            self.beta = nn.Parameter(torch.empty(output_dim), requires_grad=True)
        else:
            # Use pre-defined "membrane_threshold", "alpha" and "beta"
            self.membrane_threshold = torch.tensor([membrane_threshold], dtype=dtype, device=device)
            self.alpha = torch.tensor([alpha], dtype=dtype, device=device)
            self.beta = torch.tensor([beta], dtype=dtype, device=device)

        self.recurrent1d_layer = None
        if recurrent_flag:
            self.recurrent1d_layer = Recurrent1d(output_dim=self.output_dim)

    def multi_step_neuron(self, input_: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, List]]:

        self.nb_steps = input_.shape[2]
        self.nb_steps_truncated_bptt = int(self.nb_steps / self.truncated_bptt_ratio)

        # Synaptic current, membrane potential and output_spikes initialization
        synaptic_current, membrane_potential, output_spikes = self.states_dict.values()

        # Init the membrane potentials and output spikes record lists
        membrane_potential_records = []
        output_spikes_records = []

        input_ = input_.unbind(2)

        for t in range(self.nb_steps):

            input_t = input_[t]

            if self.recurrent1d_layer is not None:
                recurrent1d_input_t = self.recurrent1d_layer(output_spikes)

            output_spikes = self.spike_fn(membrane_potential - self.membrane_threshold)

            if self.detach_reset:
                reset = output_spikes.detach()  # Avoid backprop through the reset
            else:
                reset = output_spikes

            # Compute the new synaptic current and membrane potential: CSNN
            new_synaptic_current = self.alpha * synaptic_current + input_t

            if self.recurrent1d_layer is not None:
                new_synaptic_current = new_synaptic_current + recurrent1d_input_t

            # new_membrane_potential = self.beta * (membrane_potential + synaptic_current)
            if self.decay_input:
                new_membrane_potential = self.beta * membrane_potential + (1 - self.beta) * synaptic_current
            else:
                new_membrane_potential = self.beta * membrane_potential + synaptic_current

            # reset LIF
            new_membrane_potential = self.reset_neuron(new_membrane_potential, reset)

            if t < self.nb_steps - self.nb_steps_truncated_bptt:
                new_membrane_potential = new_membrane_potential.detach()
                output_spikes = output_spikes.detach()

            # Record the membrane potentials and output spikes of all trials and all neurons
            if self.save_mem:
                membrane_potential_records.append(new_membrane_potential)
            output_spikes_records.append(output_spikes)

            # Update the synaptic current and membrane potential
            synaptic_current = new_synaptic_current
            membrane_potential = new_membrane_potential

        # Update self.states_dict
        self.states_dict.update(synaptic_current=synaptic_current.clone().detach(),
                                membrane_potential=membrane_potential.clone().detach(),
                                output_spikes=output_spikes.clone().detach())

        if self.save_mem:
            membrane_potential_records = torch.stack(membrane_potential_records, 2)

        return torch.stack(output_spikes_records, 2), membrane_potential_records


class LI1d(MPNeuronModel):
    """Class that implements 1D-LI neurons layer.
    """

    def __init__(self, output_dim: int, truncated_bptt_ratio: int, alpha, beta,
                 train_neuron_parameters_flag: bool,
                 decay_input: bool, device, dtype) -> None:
        super(LI1d, self).__init__(truncated_bptt_ratio, decay_input, device, dtype)

        self.neuron_model = 'lif'

        self.output_dim = output_dim

        if train_neuron_parameters_flag:
            # Membrane leak coefficients
            self.alpha = nn.Parameter(torch.empty(output_dim), requires_grad=True)
            self.beta = nn.Parameter(torch.empty(output_dim), requires_grad=True)
        else:
            # Use pre-defined "alpha" and "beta"
            self.alpha = torch.tensor([alpha], dtype=dtype, device=device)
            self.beta = torch.tensor([beta], dtype=dtype, device=device)

    def multi_step_LI(self, input_: torch.Tensor) -> torch.Tensor:

        self.nb_steps = input_.shape[2]
        self.nb_steps_truncated_bptt = int(self.nb_steps / self.truncated_bptt_ratio)

        # Synaptic current, membrane potential and output_spikes initialization
        synaptic_current, membrane_potential = self.states_dict.values()

        # Init the membrane potentials record list
        membrane_potential_records = []

        input_ = input_.unbind(2)

        for t in range(self.nb_steps):

            input_t = input_[t]

            # Compute the new synaptic current and membrane potential: CSNN
            new_synaptic_current = self.alpha * synaptic_current + input_t

            # new_membrane_potential = self.beta * (membrane_potential + synaptic_current)
            if self.decay_input:
                new_membrane_potential = self.beta * membrane_potential + (1 - self.beta) * synaptic_current
            else:
                new_membrane_potential = self.beta * membrane_potential + synaptic_current

            if t < self.nb_steps - self.nb_steps_truncated_bptt:
                new_membrane_potential = new_membrane_potential.detach()

            # Record the membrane potentials and output spikes of all trials and all neurons
            membrane_potential_records.append(new_membrane_potential)

            # Update the synaptic current and membrane potential
            synaptic_current = new_synaptic_current
            membrane_potential = new_membrane_potential

        # Update self.states_dict
        self.states_dict.update(synaptic_current=synaptic_current.clone().detach(),
                                membrane_potential=membrane_potential.clone().detach())

        return torch.stack(membrane_potential_records, 2)


class LIF2d(NeuronModel):
    """Class that implements 2D-LIF neurons layer.
    """

    def __init__(self, output_dim: int, output_channels: int, truncated_bptt_ratio: int, membrane_threshold, alpha, beta,
                 train_neuron_parameters_flag: bool, spike_fn: Callable, reset_mode: str, detach_reset: bool,
                 recurrent_flag: bool, decay_input: bool, device, dtype) -> None:

        super(LIF2d, self).__init__(truncated_bptt_ratio, spike_fn, reset_mode, detach_reset, decay_input, device, dtype)

        self.neuron_model = 'lif'

        self.output_channels = output_channels
        self.output_dim = output_dim

        if train_neuron_parameters_flag:
            # Membrane threshold
            self.membrane_threshold = nn.Parameter(torch.empty(output_channels, output_dim), requires_grad=True)
            # Membrane leak coefficients
            self.alpha = nn.Parameter(torch.empty(output_channels, output_dim), requires_grad=True)
            self.beta = nn.Parameter(torch.empty(output_channels, output_dim), requires_grad=True)
        else:
            # Use pre-defined "membrane_threshold", "alpha" and "beta"
            self.membrane_threshold = torch.tensor([membrane_threshold], dtype=dtype, device=device)
            self.alpha = torch.tensor([alpha], dtype=dtype, device=device)
            self.beta = torch.tensor([beta], dtype=dtype, device=device)

        self.recurrent2d_layer = None
        if recurrent_flag:
            self.recurrent2d_layer = Recurrent2d(output_channels=self.output_channels)

    def multi_step_neuron(self, input_: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, List]]:

        self.nb_steps = input_.shape[3]
        self.nb_steps_truncated_bptt = int(self.nb_steps / self.truncated_bptt_ratio)

        # Synaptic current, membrane potential and output_spikes initialization
        synaptic_current, membrane_potential, output_spikes = self.states_dict.values()

        # Init the membrane potentials and output spikes record lists
        membrane_potential_records = []
        output_spikes_records = []

        input_ = input_.unbind(3)

        for t in range(self.nb_steps):

            input_t = input_[t]

            if self.recurrent2d_layer is not None:
                recurrent2d_input_t = self.recurrent2d_layer(output_spikes)

            output_spikes = self.spike_fn(membrane_potential - self.membrane_threshold)

            if self.detach_reset:
                reset = output_spikes.detach()  # Avoid backprop through the reset
            else:
                reset = output_spikes

            # Compute the new synaptic current and membrane potential: CSNN
            new_synaptic_current = self.alpha * synaptic_current + input_t

            if self.recurrent2d_layer is not None:
                new_synaptic_current = new_synaptic_current + recurrent2d_input_t

            # new_membrane_potential = self.beta * (membrane_potential + synaptic_current)
            if self.decay_input:
                new_membrane_potential = self.beta * membrane_potential + (1 - self.beta) * synaptic_current
            else:
                new_membrane_potential = self.beta * membrane_potential + synaptic_current

            # reset LIF
            new_membrane_potential = self.reset_neuron(new_membrane_potential, reset)

            if t < self.nb_steps - self.nb_steps_truncated_bptt:
                new_membrane_potential = new_membrane_potential.detach()
                output_spikes = output_spikes.detach()

            # Record the membrane potentials and output spikes of all trials and all neurons
            if self.save_mem:
                membrane_potential_records.append(new_membrane_potential)
            output_spikes_records.append(output_spikes)

            # Update the synaptic current and membrane potential
            synaptic_current = new_synaptic_current
            membrane_potential = new_membrane_potential

        # Update self.states_dict
        self.states_dict.update(synaptic_current=synaptic_current.clone().detach(),
                                membrane_potential=membrane_potential.clone().detach(),
                                output_spikes=output_spikes.clone().detach())

        if self.save_mem:
            membrane_potential_records = torch.stack(membrane_potential_records, 3)

        return torch.stack(output_spikes_records, 3), membrane_potential_records


class LI2d(MPNeuronModel):
    """Class that implements 2D-LI neurons layer.
    """

    def __init__(self, output_dim: int, output_channels: int, truncated_bptt_ratio: int, alpha, beta,
                 train_neuron_parameters_flag: bool, decay_input: bool, device, dtype) -> None:
        super(LI2d, self).__init__(truncated_bptt_ratio, decay_input, device, dtype)

        self.neuron_model = 'lif'

        self.output_channels = output_channels
        self.output_dim = output_dim

        if train_neuron_parameters_flag:
            # Membrane leak coefficients
            self.alpha = nn.Parameter(torch.empty(output_channels, output_dim), requires_grad=True)
            self.beta = nn.Parameter(torch.empty(output_channels, output_dim), requires_grad=True)
        else:
            # Use pre-defined "alpha" and "beta"
            self.alpha = torch.tensor([alpha], dtype=dtype, device=device)
            self.beta = torch.tensor([beta], dtype=dtype, device=device)

    def multi_step_LI(self, input_: torch.Tensor) -> torch.Tensor:

        self.nb_steps = input_.shape[3]
        self.nb_steps_truncated_bptt = int(self.nb_steps / self.truncated_bptt_ratio)

        # Synaptic current, membrane potential and output_spikes initialization
        synaptic_current, membrane_potential = self.states_dict.values()

        # Init the membrane potentials record list
        membrane_potential_records = []

        input_ = input_.unbind(3)

        for t in range(self.nb_steps):

            input_t = input_[t]

            # Compute the new synaptic current and membrane potential: CSNN
            new_synaptic_current = self.alpha * synaptic_current + input_t

            # new_membrane_potential = self.beta * (membrane_potential + synaptic_current)
            if self.decay_input:
                new_membrane_potential = self.beta * membrane_potential + (1 - self.beta) * synaptic_current
            else:
                new_membrane_potential = self.beta * membrane_potential + synaptic_current

            if t < self.nb_steps - self.nb_steps_truncated_bptt:
                new_membrane_potential = new_membrane_potential.detach()

            # Record the membrane potentials and output spikes of all trials and all neurons
            membrane_potential_records.append(new_membrane_potential)

            # Update the synaptic current and membrane potential
            synaptic_current = new_synaptic_current
            membrane_potential = new_membrane_potential

        # Update self.states_dict
        self.states_dict.update(synaptic_current=synaptic_current.clone().detach(),
                                membrane_potential=membrane_potential.clone().detach())

        return torch.stack(membrane_potential_records, 3)


class PLIF2d(NeuronModel):
    """Class that implements 2D-ParametricLIF neurons layer.

    Fang, Wei, et al. "Incorporating learnable membrane time constant to enhance learning of spiking neural networks."
    Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
    """

    def __init__(self, output_dim: int, output_channels: int, truncated_bptt_ratio: int, membrane_threshold, alpha, beta,
                 train_neuron_parameters_flag: bool, spike_fn: Callable, reset_mode: str, detach_reset: bool,
                 recurrent_flag: bool, decay_input: bool, device, dtype) -> None:

        super(PLIF2d, self).__init__(truncated_bptt_ratio, spike_fn, reset_mode, detach_reset, decay_input, device, dtype)

        self.neuron_model = 'plif'

        self.output_channels = output_channels
        self.output_dim = output_dim

        if train_neuron_parameters_flag:
            # Membrane threshold
            self.membrane_threshold = nn.Parameter(torch.empty(output_channels, output_dim), requires_grad=True)
            # Membrane leak coefficients
            self.alpha = nn.Parameter(torch.empty(output_channels, output_dim), requires_grad=True)
            self.beta = nn.Parameter(torch.empty(output_channels, output_dim), requires_grad=True)
        else:
            # Use pre-defined "membrane_threshold", "alpha" and "beta"
            self.membrane_threshold = torch.tensor([membrane_threshold], dtype=dtype, device=device)
            self.alpha = torch.tensor([alpha], dtype=dtype, device=device)
            self.beta = torch.tensor([beta], dtype=dtype, device=device)

        self.recurrent2d_layer = None
        if recurrent_flag:
            self.recurrent2d_layer = Recurrent2d(output_channels=self.output_channels)

    def multi_step_neuron(self, input_: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, List]]:

        self.nb_steps = input_.shape[3]
        self.nb_steps_truncated_bptt = int(self.nb_steps / self.truncated_bptt_ratio)

        # Synaptic current, membrane potential and output_spikes initialization
        synaptic_current, membrane_potential, output_spikes = self.states_dict.values()

        # Init the membrane potentials and output spikes record lists
        membrane_potential_records = []
        output_spikes_records = []

        input_ = input_.unbind(3)

        for t in range(self.nb_steps):

            input_t = input_[t]

            if self.recurrent2d_layer is not None:
                recurrent2d_input_t = self.recurrent2d_layer(output_spikes)

            output_spikes = self.spike_fn(membrane_potential - self.membrane_threshold)

            if self.detach_reset:
                reset = output_spikes.detach()  # Avoid backprop through the reset
            else:
                reset = output_spikes

            # Compute the new synaptic current and membrane potential: CSNN
            new_synaptic_current = self.alpha.sigmoid() * synaptic_current + input_t

            if self.recurrent2d_layer is not None:
                new_synaptic_current = new_synaptic_current + recurrent2d_input_t

            # new_membrane_potential = self.beta * (membrane_potential + synaptic_current)
            if self.decay_input:
                new_membrane_potential = self.beta * membrane_potential + (1 - self.beta) * synaptic_current
            else:
                new_membrane_potential = self.beta * membrane_potential + synaptic_current

            # reset LIF
            new_membrane_potential = self.reset_neuron(new_membrane_potential, reset)

            if t < self.nb_steps - self.nb_steps_truncated_bptt:
                new_membrane_potential = new_membrane_potential.detach()
                output_spikes = output_spikes.detach()

            # Record the membrane potentials and output spikes of all trials and all neurons
            if self.save_mem:
                membrane_potential_records.append(new_membrane_potential)
            output_spikes_records.append(output_spikes)

            # Update the synaptic current and membrane potential
            synaptic_current = new_synaptic_current
            membrane_potential = new_membrane_potential

        # Update self.states_dict
        self.states_dict.update(synaptic_current=synaptic_current.clone().detach(),
                                membrane_potential=membrane_potential.clone().detach(),
                                output_spikes=output_spikes.clone().detach())

        if self.save_mem:
            membrane_potential_records = torch.stack(membrane_potential_records, 3)

        return torch.stack(output_spikes_records, 3), membrane_potential_records


class PLI2d(MPNeuronModel):
    """Class that implements 2D-ParametricLI neurons layer.
    """

    def __init__(self, output_dim: int, output_channels: int, truncated_bptt_ratio: int, alpha, beta,
                 train_neuron_parameters_flag: bool, decay_input: bool, device, dtype) -> None:
        super(PLI2d, self).__init__(truncated_bptt_ratio, decay_input, device, dtype)

        self.neuron_model = 'plif'

        self.output_channels = output_channels
        self.output_dim = output_dim

        if train_neuron_parameters_flag:
            # Membrane leak coefficients
            self.alpha = nn.Parameter(torch.empty(output_channels, output_dim), requires_grad=True)
            self.beta = nn.Parameter(torch.empty(output_channels, output_dim), requires_grad=True)
        else:
            # Use pre-defined "alpha" and "beta"
            self.alpha = torch.tensor([alpha], dtype=dtype, device=device)
            self.beta = torch.tensor([beta], dtype=dtype, device=device)

    def multi_step_LI(self, input_: torch.Tensor) -> torch.Tensor:

        self.nb_steps = input_.shape[3]
        self.nb_steps_truncated_bptt = int(self.nb_steps / self.truncated_bptt_ratio)

        # Synaptic current, membrane potential and output_spikes initialization
        synaptic_current, membrane_potential = self.states_dict.values()

        # Init the membrane potentials record list
        membrane_potential_records = []

        input_ = input_.unbind(3)

        for t in range(self.nb_steps):

            input_t = input_[t]

            # Compute the new synaptic current and membrane potential: CSNN
            new_synaptic_current = self.alpha.sigmoid() * synaptic_current + input_t

            # new_membrane_potential = self.beta * (membrane_potential + synaptic_current)
            if self.decay_input:
                new_membrane_potential = self.beta * membrane_potential + (1 - self.beta) * synaptic_current
            else:
                new_membrane_potential = self.beta * membrane_potential + synaptic_current

            if t < self.nb_steps - self.nb_steps_truncated_bptt:
                new_membrane_potential = new_membrane_potential.detach()

            # Record the membrane potentials and output spikes of all trials and all neurons
            membrane_potential_records.append(new_membrane_potential)

            # Update the synaptic current and membrane potential
            synaptic_current = new_synaptic_current
            membrane_potential = new_membrane_potential

        # Update self.states_dict
        self.states_dict.update(synaptic_current=synaptic_current.clone().detach(),
                                membrane_potential=membrane_potential.clone().detach())

        return torch.stack(membrane_potential_records, 3)


class KLIF2d(NeuronModel):
    """Class that implements 2D-KLIF neurons layer.

    Jiang, Chunming, and Yilei Zhang. "KLIF: An optimized spiking neuron unit for tuning surrogate gradient slope and
    membrane potential." arXiv preprint arXiv:2302.09238 (2023).
    """

    def __init__(self, output_dim: int, output_channels: int, truncated_bptt_ratio: int, membrane_threshold, alpha, beta, k,
                 train_neuron_parameters_flag: bool, spike_fn: Callable, reset_mode: str, detach_reset: bool,
                 recurrent_flag: bool, decay_input: bool, device, dtype) -> None:

        super(KLIF2d, self).__init__(truncated_bptt_ratio, spike_fn, reset_mode, detach_reset, decay_input, device, dtype)

        self.neuron_model = 'klif'

        self.output_channels = output_channels
        self.output_dim = output_dim

        if train_neuron_parameters_flag:
            # Membrane threshold
            self.membrane_threshold = nn.Parameter(torch.empty(output_channels, output_dim), requires_grad=True)
            # Membrane leak coefficients
            self.alpha = nn.Parameter(torch.empty(output_channels, output_dim), requires_grad=True)
            self.beta = nn.Parameter(torch.empty(output_channels, output_dim), requires_grad=True)
            self.k = nn.Parameter(torch.empty(output_channels, output_dim), requires_grad=True)
        else:
            # Use pre-defined "membrane_threshold", "alpha" and "beta"
            self.membrane_threshold = torch.tensor([membrane_threshold], dtype=dtype, device=device)
            self.alpha = torch.tensor([alpha], dtype=dtype, device=device)
            self.beta = torch.tensor([beta], dtype=dtype, device=device)
            self.k = torch.tensor([k], dtype=dtype, device=device)

        self.recurrent2d_layer = None
        if recurrent_flag:
            self.recurrent2d_layer = Recurrent2d(output_channels=self.output_channels)

    def multi_step_neuron(self, input_: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, List]]:

        self.nb_steps = input_.shape[3]
        self.nb_steps_truncated_bptt = int(self.nb_steps / self.truncated_bptt_ratio)

        # Synaptic current, membrane potential and output_spikes initialization
        synaptic_current, membrane_potential, output_spikes = self.states_dict.values()

        # Init the membrane potentials and output spikes record lists
        membrane_potential_records = []
        output_spikes_records = []

        input_ = input_.unbind(3)

        for t in range(self.nb_steps):

            input_t = input_[t]

            if self.recurrent2d_layer is not None:
                recurrent2d_input_t = self.recurrent2d_layer(output_spikes)

            membrane_potential = torch.relu_(self.k * membrane_potential)

            output_spikes = self.spike_fn(membrane_potential - self.membrane_threshold)

            if self.detach_reset:
                reset = output_spikes.detach()  # Avoid backprop through the reset
            else:
                reset = output_spikes

            # Compute the new synaptic current and membrane potential: CSNN
            new_synaptic_current = self.alpha * synaptic_current + input_t

            if self.recurrent2d_layer is not None:
                new_synaptic_current = new_synaptic_current + recurrent2d_input_t

            # new_membrane_potential = self.beta * (membrane_potential + synaptic_current)
            if self.decay_input:
                new_membrane_potential = self.beta * membrane_potential + (1 - self.beta) * synaptic_current
            else:
                new_membrane_potential = self.beta * membrane_potential + synaptic_current

            # reset LIF
            new_membrane_potential = self.reset_neuron(new_membrane_potential, reset)

            if t < self.nb_steps - self.nb_steps_truncated_bptt:
                new_membrane_potential = new_membrane_potential.detach()
                output_spikes = output_spikes.detach()

            # Record the membrane potentials and output spikes of all trials and all neurons
            if self.save_mem:
                membrane_potential_records.append(new_membrane_potential)
            output_spikes_records.append(output_spikes)

            # Update the synaptic current and membrane potential
            synaptic_current = new_synaptic_current
            membrane_potential = new_membrane_potential

        # Update self.states_dict
        self.states_dict.update(synaptic_current=synaptic_current.clone().detach(),
                                membrane_potential=membrane_potential.clone().detach(),
                                output_spikes=output_spikes.clone().detach())

        if self.save_mem:
            membrane_potential_records = torch.stack(membrane_potential_records, 3)

        return torch.stack(output_spikes_records, 3), membrane_potential_records


class KLI2d(MPNeuronModel):
    """Class that implements 2D-KLI neurons layer.
    """

    def __init__(self, output_dim: int, output_channels: int, truncated_bptt_ratio: int, alpha, beta, k,
                 train_neuron_parameters_flag: bool, decay_input: bool, device, dtype) -> None:
        super(KLI2d, self).__init__(truncated_bptt_ratio, decay_input, device, dtype)

        self.neuron_model = 'klif'

        self.output_channels = output_channels
        self.output_dim = output_dim

        if train_neuron_parameters_flag:
            # Membrane leak coefficients
            self.alpha = nn.Parameter(torch.empty(output_channels, output_dim), requires_grad=True)
            self.beta = nn.Parameter(torch.empty(output_channels, output_dim), requires_grad=True)
            self.k = nn.Parameter(torch.empty(output_channels, output_dim), requires_grad=True)
        else:
            # Use pre-defined "alpha" and "beta"
            self.alpha = torch.tensor([alpha], dtype=dtype, device=device)
            self.beta = torch.tensor([beta], dtype=dtype, device=device)
            self.k = torch.tensor([k], dtype=dtype, device=device)

    def multi_step_LI(self, input_: torch.Tensor) -> torch.Tensor:

        self.nb_steps = input_.shape[3]
        self.nb_steps_truncated_bptt = int(self.nb_steps / self.truncated_bptt_ratio)

        # Synaptic current, membrane potential and output_spikes initialization
        synaptic_current, membrane_potential = self.states_dict.values()

        # Init the membrane potentials record list
        membrane_potential_records = []

        input_ = input_.unbind(3)

        for t in range(self.nb_steps):

            input_t = input_[t]

            membrane_potential = torch.relu_(self.k * membrane_potential)

            # Compute the new synaptic current and membrane potential: CSNN
            new_synaptic_current = self.alpha * synaptic_current + input_t

            # new_membrane_potential = self.beta * (membrane_potential + synaptic_current)
            if self.decay_input:
                new_membrane_potential = self.beta * membrane_potential + (1 - self.beta) * synaptic_current
            else:
                new_membrane_potential = self.beta * membrane_potential + synaptic_current

            if t < self.nb_steps - self.nb_steps_truncated_bptt:
                new_membrane_potential = new_membrane_potential.detach()

            # Record the membrane potentials and output spikes of all trials and all neurons
            membrane_potential_records.append(new_membrane_potential)

            # Update the synaptic current and membrane potential
            synaptic_current = new_synaptic_current
            membrane_potential = new_membrane_potential

        # Update self.states_dict
        self.states_dict.update(synaptic_current=synaptic_current.clone().detach(),
                                membrane_potential=membrane_potential.clone().detach())

        return torch.stack(membrane_potential_records, 3)


class IF2d(NeuronModel):
    """Class that implements 2D-IF neurons layer.
    """

    def __init__(self, output_dim: int, output_channels: int, truncated_bptt_ratio: int, membrane_threshold,
                 train_neuron_parameters_flag: bool, spike_fn: Callable, reset_mode: str, detach_reset: bool,
                 recurrent_flag: bool, decay_input: bool, device, dtype) -> None:

        super(IF2d, self).__init__(truncated_bptt_ratio, spike_fn, reset_mode, detach_reset, decay_input, device, dtype)

        self.neuron_model = 'if'

        self.output_channels = output_channels
        self.output_dim = output_dim

        if train_neuron_parameters_flag:
            # Membrane threshold
            self.membrane_threshold = nn.Parameter(torch.empty(output_channels), requires_grad=True)
        else:
            # Use pre-defined "membrane_threshold"
            self.membrane_threshold = membrane_threshold

        self.recurrent2d_layer = None
        if recurrent_flag:
            self.recurrent2d_layer = Recurrent2d(output_channels=self.output_channels)

    def multi_step_neuron(self, input_: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, List]]:

        self.nb_steps = input_.shape[3]
        self.nb_steps_truncated_bptt = int(self.nb_steps / self.truncated_bptt_ratio)

        # Synaptic current, membrane potential and output_spikes initialization
        synaptic_current, membrane_potential, output_spikes = self.states_dict.values()

        # Init the membrane potentials and output spikes record lists
        membrane_potential_records = []
        output_spikes_records = []

        input_ = input_.unbind(3)

        for t in range(self.nb_steps):

            input_t = input_[t]

            if self.recurrent2d_layer is not None:
                recurrent2d_input_t = self.recurrent2d_layer(output_spikes)

            output_spikes = self.spike_fn(membrane_potential - self.membrane_threshold)

            if self.detach_reset:
                reset = output_spikes.detach()  # Avoid backprop through the reset
            else:
                reset = output_spikes

            # Compute the new synaptic current and membrane potential: CSNN
            new_synaptic_current = synaptic_current + input_t

            if self.recurrent2d_layer is not None:
                new_synaptic_current = new_synaptic_current + recurrent2d_input_t

            new_membrane_potential = membrane_potential + synaptic_current

            # reset LIF
            new_membrane_potential = self.reset_neuron(new_membrane_potential, reset)

            if t < self.nb_steps - self.nb_steps_truncated_bptt:
                new_membrane_potential = new_membrane_potential.detach()
                output_spikes = output_spikes.detach()

            # Record the membrane potentials and output spikes of all trials and all neurons
            if self.save_mem:
                membrane_potential_records.append(new_membrane_potential)
            output_spikes_records.append(output_spikes)

            # Update the synaptic current and membrane potential
            synaptic_current = new_synaptic_current
            membrane_potential = new_membrane_potential

        # Update self.states_dict
        self.states_dict.update(synaptic_current=synaptic_current.clone().detach(),
                                membrane_potential=membrane_potential.clone().detach(),
                                output_spikes=output_spikes.clone().detach())

        if self.save_mem:
            membrane_potential_records = torch.stack(membrane_potential_records, 3)

        return torch.stack(output_spikes_records, 3), membrane_potential_records


class I2d(MPNeuronModel):
    """Class that implements 2D-I neurons layer.
    """

    def __init__(self, output_dim: int, output_channels: int, truncated_bptt_ratio: int, decay_input: bool,
                 device, dtype) -> None:

        super(I2d, self).__init__(truncated_bptt_ratio, decay_input, device, dtype)

        self.neuron_model = 'if'

        self.output_channels = output_channels
        self.output_dim = output_dim

    def multi_step_LI(self, input_: torch.Tensor) -> torch.Tensor:

        self.nb_steps = input_.shape[3]
        self.nb_steps_truncated_bptt = int(self.nb_steps / self.truncated_bptt_ratio)

        # Synaptic current, membrane potential and output_spikes initialization
        synaptic_current, membrane_potential = self.states_dict.values()

        # Init the membrane potentials record list
        membrane_potential_records = []

        input_ = input_.unbind(3)

        for t in range(self.nb_steps):

            input_t = input_[t]

            # Compute the new synaptic current and membrane potential: CSNN
            new_synaptic_current = synaptic_current + input_t

            new_membrane_potential = membrane_potential + synaptic_current

            if t < self.nb_steps - self.nb_steps_truncated_bptt:
                new_membrane_potential = new_membrane_potential.detach()

            # Record the membrane potentials and output spikes of all trials and all neurons
            membrane_potential_records.append(new_membrane_potential)

            # Update the synaptic current and membrane potential
            synaptic_current = new_synaptic_current
            membrane_potential = new_membrane_potential

        # Update self.states_dict
        self.states_dict.update(synaptic_current=synaptic_current.clone().detach(),
                                membrane_potential=membrane_potential.clone().detach())

        return torch.stack(membrane_potential_records, 3)
