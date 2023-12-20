"""
File:
    model/TrainValidTestManager.py

Description:
    Defines the TrainValidTestManager class.
"""

import os
from src.model.utils import pad_coefficients, complex_spectrogram
from typing import Optional, List, Tuple
import numpy as np
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.optim import ZeroRedundancyOptimizer
from collections import OrderedDict
from pathlib import Path
from typing import Union
from tqdm import tqdm
from comet_ml import Experiment
from torch.utils.data import DataLoader
from src.model.SpikingModel import FCSNN, CSNN, UNetSNN, ResBottleneckUNetSNN
from src.visualization.VisualizationManager import VisualizationManager
from src.model.LossManager import LossManager
from src.data.TransformManager import TransformManager
from src.data.constants import RepresentationName


class TrainValidTestManager:
    """Class that implements the training, validation and testing functions.
    """

    def __init__(self,
                 model: Union[FCSNN, CSNN, UNetSNN, ResBottleneckUNetSNN],
                 model_name: str,
                 model_file_dir: Union[str, Path],
                 exp_model_file_dir: Union[str, Path],
                 use_mask: bool,
                 dataloader_manager: DataLoader,
                 dataloader_manager_valid: Optional[DataLoader],
                 representation_name: str,
                 transform_manager: TransformManager,
                 visualization_manager: VisualizationManager,
                 weights: Optional[torch.Tensor],
                 loss_name: List[str],
                 loss_weight: List[float],
                 loss_bias: List[float],
                 optimizer_name: str,
                 learning_rate: float,
                 betas: tuple,
                 scheduler_name: Optional[str],
                 lr_scheduler_max_lr: float,
                 lr_scheduler_gamma: float,
                 clip_type: Optional[str],
                 clip_value: float,
                 nb_epochs: int,
                 nb_warmup_epochs: int,
                 nb_steps: int,
                 nb_steps_bin: int,
                 experiment: Optional[Experiment] = None,
                 save_mem: bool = True,
                 save_model: bool = False,
                 train_neuron_parameters: bool = False,
                 log_lr: bool = True,
                 log_batch_loss: bool = False,
                 log_grad_norm: bool = False,
                 log_hist_flag: bool = False,
                 use_ddp: bool = False,
                 device: torch.device = 'cpu',
                 dtype: torch.dtype = torch.float,
                 evaluate_flag: bool = False,
                 use_amp: bool = True,
                 use_zero: bool = True,
                 weight_decay: float = 0.,
                 amsgrad_flag: bool = False,
                 empty_cache: bool = False,
                 pretrained_flag: bool = False,
                 perceptual_metric_flag: bool = False,
                 perceptual_metric_name: str = 'stoi',
                 nb_digits: int = 5
                 ) -> None:

        self.representation_name = representation_name
        self.transform_manager = transform_manager

        self.use_ddp = use_ddp
        self.all_reduce_flag = True

        if use_ddp:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.device = device
        self.dtype = dtype

        self.dataloader_manager = dataloader_manager
        self.dataloader_manager_valid = dataloader_manager_valid

        self.visualization_manager = visualization_manager

        self.weights = weights

        self.perceptual_metric_flag_train = False
        self.perceptual_metric_flag = perceptual_metric_flag
        self.perceptual_metric_name = perceptual_metric_name

        self.evaluate_flag = evaluate_flag

        # Set training epochs value
        self.nb_epochs = nb_epochs

        # Initialize pretrained epochs value
        self.nb_pretrained_epochs = 0
        # Initialize warmup epochs value
        self.nb_warmup_epochs = nb_warmup_epochs

        # Get model file name
        self.model_file_dir = model_file_dir
        self.exp_model_file_dir = exp_model_file_dir

        # Get model name
        self.model_name = model_name

        # Get model
        self.model = model

        # Get SE approach
        self.use_mask = use_mask

        self.snn_flag = False
        if use_ddp:
            if hasattr(self.model.module, 'init_state'):
                self.snn_flag = True
        else:
            if hasattr(self.model, 'init_state'):
                self.snn_flag = True

        # Find which parameters to train
        self.learning_rate = learning_rate

        # Define optimizer
        self.use_zero = use_zero
        self.optimizer_name = optimizer_name
        self.optimizer = None
        self.prepare_optimizer(betas, weight_decay, amsgrad_flag)

        # To perform batch sequence length
        self.nb_steps = nb_steps
        self.nb_steps_bin = nb_steps_bin
        self.nb_chunks = 1
        if nb_steps_bin != nb_steps:
            self.nb_chunks = nb_steps // nb_steps_bin

        # Define scheduler
        self.scheduler_name = scheduler_name
        self.scheduler = None
        self.prepare_scheduler(lr_scheduler_max_lr, lr_scheduler_gamma)

        # Gradient clip
        self.clip_type = clip_type
        self.clip_value = clip_value

        # Define GradScaler
        self.use_amp = use_amp
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Define loss function
        self.loss_fn = LossManager(representation_name=representation_name,
                                   loss_name=loss_name,
                                   loss_weight=loss_weight,
                                   loss_bias=loss_bias,
                                   metric_name=perceptual_metric_name,
                                   weights=weights,
                                   transform_manager=transform_manager,
                                   ).cuda()

        # Define round digits
        self.nb_digits = nb_digits

        # Initialize results dictionary
        self.results = {'Training loss': [], f'Training {self.perceptual_metric_name}': [],
                        'Validation loss': [], f'Validation {self.perceptual_metric_name}': [],
                        'Test loss': [], f'Test {self.perceptual_metric_name}': []}

        self.empty_cache = empty_cache

        # Load checkpoint if pretrained_flag is True
        self.pretrained_flag = pretrained_flag
        if self.pretrained_flag:
            self.load_checkpoint(filepath=self.model_file_dir)

        # Define experiment
        self.experiment = experiment
        self.save_mem = save_mem
        self.save_model = save_model
        self.train_neuron_parameters = train_neuron_parameters
        self.log_lr = log_lr
        self.log_batch_loss = log_batch_loss
        self.log_grad_norm = log_grad_norm
        self.log_hist_flag = log_hist_flag

        if self.empty_cache:
            self._empty_cache()

    def train_model(self) -> None:
        """Method that trains the model and saves the trained model.
        """

        # Declare tqdm progress bar
        progress_bar = tqdm(total=len(self.dataloader_manager), leave=False, desc=f'Epoch [0/{self.nb_epochs - 1}]',
                            postfix='Training Loss: 0.')

        # Log model parameters
        if self.experiment and self.log_hist_flag:
            if self.rank == 0:
                self._experiment_param_hist(epoch=0)
            if self.use_ddp:
                dist.barrier()

        # Update nb_epochs
        if self.pretrained_flag:
            self.nb_epochs = self.nb_epochs - self.nb_pretrained_epochs

        if self.nb_warmup_epochs > 0 and (not self.pretrained_flag):
            nb_warmup_steps = len(self.dataloader_manager) * self.nb_warmup_epochs
            current_warmup_step = 1
            for param in self.optimizer.param_groups:
                param['lr'] /= nb_warmup_steps

        step = 0

        # The training loop
        for epoch in range(self.nb_epochs):

            # Specify that the model will be trained
            if self.use_ddp:
                self.model.module.train()
            else:
                self.model.train()

            # Init loss/rec error
            train_loss_mean = 0.
            train_perceptual_metric_mean = 0.

            if self.use_ddp:
                self.dataloader_manager.sampler.set_epoch(epoch + self.nb_pretrained_epochs)

            if self.log_grad_norm:
                grad_norm_dict = self._log_grad_norm()

            if self.empty_cache:
                self._empty_cache()

            # Loop through each mini-batch from the data_files loader
            for batch_index, (x_data, y_data, index) in enumerate(self.dataloader_manager):

                batch_loss_mean = 0.
                mem_chunks = []

                # Load data to GPU
                x_data = x_data.to(device=self.device, non_blocking=True)
                y_data = y_data.to(device=self.device, non_blocking=True)

                x_data, y_data, x_data_phase, y_data_phase = self._preprocess_data(x_data, y_data)

                if self.transform_manager:
                    x_data, y_data = self.transform_manager(x_data), self.transform_manager(y_data)

                x_data_chunks, y_data_chunks, x_data_phase_chunks, y_data_phase_chunks = self._data_to_chunks(
                    x_data, y_data, x_data_phase, y_data_phase)

                self._init_state(x_data.shape[0])

                for k in range(self.nb_chunks):

                    if self.use_amp:
                        # Runs the forward pass with autocasting.
                        with torch.cuda.amp.autocast():
                            # Perform a forward pass
                            mem, output = self.model(x_data_chunks[k])
                            if self.use_mask:
                                mask = torch.relu(mem)
                                mem = x_data_chunks[k] * mask
                                for i in range(len(output)):
                                    mask_i = torch.relu(output[i])
                                    output[i] = x_data_chunks[k] * mask_i
                            mem_chunks.append(mem.clone().detach())

                            # Compute the loss
                            loss = self.loss_fn(mem.view(y_data_chunks[k].shape), y_data_chunks[k],
                                                x_data_phase_chunks[k], y_data_phase_chunks[k])
                            for j in range(len(output)):
                                loss = loss + self.loss_fn(output[j], y_data_chunks[k],
                                                           x_data_phase_chunks[k], y_data_phase_chunks[k])

                        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                        # Backward passes under autocast are not recommended.
                        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                        self.scaler.scale(loss).backward()

                        batch_loss_mean += float(loss)

                        # Get gradient norms
                        if self.log_grad_norm:
                            # self.scaler.unscale_(self.optimizer)
                            grad_norm_dict = self._log_grad_norm(grad_norm_dict)

                        # Unscales the gradients of optimizer's assigned params in-place
                        # self.scaler.unscale_(self.optimizer)

                        if self.clip_type is not None:
                            self._clip_grad()

                        # print(f'\t - scaler.scale = {self.scaler.get_scale()}')

                        # Unscale the gradients of optimizer's assigned params in-place
                        # scaler.step() first unscales the gradients of the optimizer's assigned params.
                        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                        # otherwise, optimizer.step() is skipped.
                        self.scaler.step(self.optimizer)

                        # Updates the scale for next iteration.
                        self.scaler.update()

                    else:
                        # Perform a forward pass
                        mem, output = self.model(x_data_chunks[k])
                        if self.use_mask:
                            mask = torch.relu(mem)
                            mem = x_data_chunks[k] * mask
                            for i in range(len(output)):
                                mask_i = torch.relu(output[i])
                                output[i] = x_data_chunks[k] * mask_i
                        mem_chunks.append(mem.clone().detach())

                        # Compute the loss
                        loss = self.loss_fn(mem.view(y_data_chunks[k].shape), y_data_chunks[k],
                                            x_data_phase_chunks[k], y_data_phase_chunks[k])
                        for j in range(len(output)):
                            loss = loss + self.loss_fn(output[j], y_data_chunks[k],
                                                       x_data_phase_chunks[k], y_data_phase_chunks[k])

                        # Perform a backward pass
                        # (Compute gradient of the loss with respect to all the learnable parameters)
                        loss.backward()

                        batch_loss_mean += float(loss)

                        # Get gradient norms
                        if self.log_grad_norm:
                            grad_norm_dict = self._log_grad_norm(grad_norm_dict)

                        if self.clip_type is not None:
                            self._clip_grad()

                        # Update weights with respect to the current gradient
                        self.optimizer.step()

                    # Reset all gradients to zero (Gradients accumulate by default)
                    self.optimizer.zero_grad(set_to_none=True)

                    step += 1

                    # self._find_unused_params()

                    if self.snn_flag:
                        # Clamp SNN neuron parameters (to avid negative values, otherwise, loss == nan)
                        if self.train_neuron_parameters:
                            self._clamp_neuron_params()

                    self._init_rec()

                    if self.empty_cache:
                        self._empty_cache()

                # Log learning rate to Comet.ml
                if self.experiment and self.log_lr:
                    for param in self.optimizer.param_groups:
                        self.experiment.log_metric('batch_learning_rate', param['lr'], step=step)

                # Update learning rate during warm up epochs
                if epoch < self.nb_warmup_epochs and (not self.pretrained_flag):
                    # # Log batch_learning_rate
                    # if self.experiment:
                    #     for param in self.optimizer.param_groups:
                    #         self.experiment.log_metric('warmup_batch_learning_rate', param['lr'],
                    #                                    step=(epoch * len(self.dataloader_manager)) + batch_index)

                    # Update batch_learning_rate
                    warmp_factor = (1 + current_warmup_step) / current_warmup_step
                    for param in self.optimizer.param_groups:
                        param['lr'] *= warmp_factor
                    current_warmup_step += 1

                # Add the current loss
                batch_loss_mean /= self.nb_chunks
                train_loss_mean += batch_loss_mean

                if self.perceptual_metric_flag_train:
                    x_data_phase_chunks_ = None
                    if self.representation_name == RepresentationName.stft:
                        x_data_phase_chunks_ = torch.cat(x_data_phase_chunks, -1)

                    train_perceptual_metric = self.loss_fn.perceptual_metric(torch.cat(mem_chunks, -1),
                                                                             y_data,
                                                                             x_data_phase_chunks_,
                                                                             y_data_phase)
                    train_perceptual_metric_mean += train_perceptual_metric

                # Log batch loss to Comet.ml; step is each batch
                if self.log_batch_loss and self.experiment:
                    self.experiment.log_metric('train_batch_loss', batch_loss_mean, step=step)

                # Update progress bar
                progress_bar.set_description_str(f'Epoch [{epoch}/{self.nb_epochs - 1}]')
                progress_bar.set_postfix_str(f'Training Loss: {batch_loss_mean:.5f}')
                progress_bar.update()

                if self.empty_cache:
                    self._empty_cache()

            # Log grad norm to Comet.ml
            if self.experiment and self.log_grad_norm:
                for index, (name, param) in enumerate(self.model.named_parameters()):
                    self.experiment.log_metric(f'batch_grad_norm__{name}', np.mean(grad_norm_dict[name]), step=epoch)

            # Log learning rate to Comet.ml
            if self.experiment and self.log_lr:
                for param in self.optimizer.param_groups:
                    self.experiment.log_metric('learning_rate', param['lr'], step=epoch)

            # Reset progress bar after epoch completion
            progress_bar.reset()

            # Compute mean loss over all batches
            train_loss_mean /= len(self.dataloader_manager)
            if self.all_reduce_flag:
                train_loss_mean = self._all_reduce_fn(train_loss_mean)

            if self.perceptual_metric_flag_train:
                train_perceptual_metric_mean /= len(self.dataloader_manager)
                if self.all_reduce_flag:
                    train_perceptual_metric_mean = self._all_reduce_fn(train_perceptual_metric_mean)

            # Print the training loss
            txt = '\nEpoch %i: Training loss = %.5f' % (epoch, float(train_loss_mean))
            if self.perceptual_metric_flag_train:
                txt += ' - Training %s = %.5f' % (self.perceptual_metric_name, float(train_perceptual_metric_mean))
            print(txt)

            # Save the training loss
            self.results['Training loss'].append(np.round(train_loss_mean, self.nb_digits))

            if self.perceptual_metric_flag_train:
                self.results[f'Training {self.perceptual_metric_name}'].append(np.round(train_perceptual_metric_mean,
                                                                                        self.nb_digits))

            # Log epoch loss to Comet.ml; step is each epoch
            if self.experiment:
                self.experiment.log_metric('train_mean_loss', train_loss_mean, step=epoch)
                if self.perceptual_metric_flag_train:
                    self.experiment.log_metric(f'train_mean_{self.perceptual_metric_name}',
                                               self.results[f'Training {self.perceptual_metric_name}'][-1], step=epoch)

            # Validate the model
            if self.evaluate_flag:
                self.validate_model(epoch)

            # Update learning rate using scheduler
            if self.scheduler:
                self.scheduler.step()

            # Log model parameters
            if self.experiment and self.log_hist_flag:
                if self.rank == 0:
                    self._experiment_param_hist(epoch=epoch + 1)
                if self.use_ddp:
                    dist.barrier()

            # Save the trained model
            if self.save_model and self.rank == 0:
                self.save_checkpoint(epoch=self.nb_pretrained_epochs + epoch + 1,
                                     filepath=self.exp_model_file_dir)

        # Close the tqdm bar
        progress_bar.close()

    def validate_model(self, epoch: int) -> None:
        """Method that evaluates the model using the validation set.

        Parameters
        ----------
        epoch: int
            Training iteration.
        """

        # Declare tqdm progress bar
        progress_bar = tqdm(total=len(self.dataloader_manager_valid), leave=False, desc='Valid',
                            postfix='Validation Loss: 0')

        # Specify that the model will be evaluated
        if self.use_ddp:
            self.model.module.eval()
        else:
            self.model.eval()

        # Create empty loss list
        valid_loss_mean = 0.
        valid_perceptual_metric_mean = 0.

        if self.save_mem and self.snn_flag:
            self._save_mem(self.save_mem)

        # Deactivate the autograd engine
        with torch.no_grad():
            for batch_index, (x_data, y_data, index) in enumerate(self.dataloader_manager_valid):

                batch_loss_mean = 0.
                mem_chunks = []

                # Load data to GPU
                x_data = x_data.to(device=self.device, non_blocking=True)
                y_data = y_data.to(device=self.device, non_blocking=True)

                x_data, y_data, x_data_phase, y_data_phase = self._preprocess_data(x_data, y_data)

                if self.transform_manager:
                    x_data, y_data = self.transform_manager(x_data), self.transform_manager(y_data)

                x_data_chunks, y_data_chunks, x_data_phase_chunks, y_data_phase_chunks = self._data_to_chunks(
                    x_data, y_data, x_data_phase, y_data_phase)

                self._init_state(x_data.shape[0])

                for k in range(self.nb_chunks):
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            # Perform a forward pass
                            mem, output = self.model(x_data_chunks[k])
                            if self.use_mask:
                                mask = torch.relu(mem)
                                mem = x_data_chunks[k] * mask
                                for i in range(len(output)):
                                    mask_i = torch.relu(output[i])
                                    output[i] = x_data_chunks[k] * mask_i
                            mem_chunks.append(mem.clone().detach())

                            # Compute the loss
                            loss = self.loss_fn(mem.view(y_data_chunks[k].shape), y_data_chunks[k],
                                                x_data_phase_chunks[k], y_data_phase_chunks[k])
                            for j in range(len(output)):
                                loss = loss + self.loss_fn(output[j], y_data_chunks[k],
                                                           x_data_phase_chunks[k], y_data_phase_chunks[k])

                            batch_loss_mean += float(loss)
                    else:
                        # Perform a forward pass
                        mem, output = self.model(x_data_chunks[k])
                        if self.use_mask:
                            mask = torch.relu(mem)
                            mem = x_data_chunks[k] * mask
                            for i in range(len(output)):
                                mask_i = torch.relu(output[i])
                                output[i] = x_data_chunks[k] * mask_i
                        mem_chunks.append(mem.clone().detach())

                        # Compute the loss
                        loss = self.loss_fn(mem.view(y_data_chunks[k].shape), y_data_chunks[k],
                                            x_data_phase_chunks[k], y_data_phase_chunks[k])
                        for j in range(len(output)):
                            loss = loss + self.loss_fn(output[j], y_data_chunks[k],
                                                       x_data_phase_chunks[k], y_data_phase_chunks[k])

                        batch_loss_mean += float(loss)

                    if k < self.nb_chunks - 1:
                        self._init_rec()

                # Add the current loss
                batch_loss_mean /= self.nb_chunks
                valid_loss_mean += batch_loss_mean

                # Update progress bar
                progress_bar.set_postfix_str(f'Validation Loss: {batch_loss_mean:.5f}')
                progress_bar.update()

                if self.perceptual_metric_flag:
                    x_data_phase_chunks_ = None
                    if self.representation_name == RepresentationName.stft:
                        x_data_phase_chunks_ = torch.cat(x_data_phase_chunks, -1)

                    valid_perceptual_metric = self.loss_fn.perceptual_metric(torch.cat(mem_chunks, -1),
                                                                             y_data,
                                                                             x_data_phase_chunks_,
                                                                             y_data_phase)
                    valid_perceptual_metric_mean += valid_perceptual_metric

                index_list = [0]
                if self.experiment and self.save_mem and batch_index in index_list:
                    self._visualize_output(torch.cat(mem_chunks, -1), torch.cat(x_data_chunks, -1),
                                           torch.cat(y_data_chunks, -1), output, epoch)
                    self._save_mem(False)

                self._init_rec()

            if self.empty_cache:
                self._empty_cache()

        # Compute mean loss over all batches
        valid_loss_mean /= len(self.dataloader_manager_valid)

        # Compute global loss (average loss values across all processes)
        if self.all_reduce_flag:
            valid_loss_mean = self._all_reduce_fn(valid_loss_mean)

        if self.perceptual_metric_flag:
            valid_perceptual_metric_mean /= len(self.dataloader_manager_valid)
            if self.all_reduce_flag:
                valid_perceptual_metric_mean = self._all_reduce_fn(valid_perceptual_metric_mean)

        # Print validation mean loss
        txt = '         Validation loss = %.5f' % valid_loss_mean
        if self.perceptual_metric_flag:
            txt += ' - Validation %s = %.5f' % (self.perceptual_metric_name, float(valid_perceptual_metric_mean))
        print(txt)

        # Save mean loss and mean accuracy in the object
        self.results['Validation loss'].append(np.round(valid_loss_mean, self.nb_digits))

        if self.perceptual_metric_flag:
            self.results[f'Validation {self.perceptual_metric_name}'].append(np.round(valid_perceptual_metric_mean, self.nb_digits))

        # Log epoch loss to Comet.ml; step is each epoch
        if self.experiment:
            self.experiment.log_metric('valid_mean_loss', valid_loss_mean, step=epoch)
            if self.perceptual_metric_flag:
                self.experiment.log_metric(f'valid_mean_{self.perceptual_metric_name}',
                                           self.results[f'Validation {self.perceptual_metric_name}'][-1], step=epoch)

        # Close the tqdm bar
        progress_bar.close()

    def test_model(self, tensor_enhanced_coefficients_dir: str, coefficients_filename: str,
                   reconstruction_test: bool = False) -> None:
        """Method that evaluates the model using the test set.

        Parameters
        ----------
        tensor_enhanced_coefficients_dir: str
            Enhanced encoded data directory.
        coefficients_filename: str
            Enhanced encoded data filename.
        reconstruction_test: str
            Boolean that indicates weather to run a reconstruction test (output_tensor = target_tensor).
        """

        # Declare tqdm progress bar
        progress_bar = tqdm(total=len(self.dataloader_manager), leave=False, desc='Test', postfix='Testing Loss: 0')

        # Log model parameters
        if self.experiment and self.log_hist_flag:
            self._experiment_param_hist(epoch=0)

        # Specify that the model will be evaluated
        if self.use_ddp:
            self.model.module.eval()
        else:
            self.model.eval()

        # Init loss
        test_loss_mean = 0.

        if self.save_mem and self.snn_flag:
            self._save_mem(self.save_mem)

        # Deactivate the autograd engine
        with torch.no_grad():
            for batch_index, (x_data, y_data, index) in enumerate(self.dataloader_manager):

                batch_loss_mean = 0.
                mem_chunks = []

                # Load data to GPU
                x_data = x_data.to(device=self.device, non_blocking=True)
                y_data = y_data.to(device=self.device, non_blocking=True)

                x_data, y_data, x_data_phase, y_data_phase = self._preprocess_data(x_data, y_data)

                if self.transform_manager:
                    x_data, y_data = self.transform_manager(x_data), self.transform_manager(y_data)

                x_data_chunks, y_data_chunks, x_data_phase_chunks, y_data_phase_chunks = self._data_to_chunks(
                    x_data, y_data, x_data_phase, y_data_phase)

                self._init_state(x_data.shape[0])

                if reconstruction_test:
                    x_data_phase_chunks = [None] * len(y_data_phase_chunks)

                for k in range(self.nb_chunks):
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            # Perform a forward pass
                            if not reconstruction_test:
                                mem, output = self.model(x_data_chunks[k])
                                if self.use_mask:
                                    mask = torch.relu(mem)
                                    mem = x_data_chunks[k] * mask
                                    for i in range(len(output)):
                                        mask_i = torch.relu(output[i])
                                        output[i] = x_data_chunks[k] * mask_i
                                mem_chunks.append(mem.clone().detach())

                            else:
                                # Reconstruction test
                                mem = torch.clone(y_data_chunks[k])
                                mem_chunks.append(mem.clone().detach())

                            # Compute the loss
                            loss = self.loss_fn(mem.view(y_data_chunks[k].shape), y_data_chunks[k],
                                                x_data_phase_chunks[k], y_data_phase_chunks[k])
                            for j in range(len(output)):
                                loss = loss + self.loss_fn(output[j], y_data_chunks[k],
                                                           x_data_phase_chunks[k], y_data_phase_chunks[k])

                            batch_loss_mean += float(loss)
                    else:
                        # Perform a forward pass
                        if not reconstruction_test:
                            mem, output = self.model(x_data_chunks[k])
                            if self.use_mask:
                                mask = torch.relu(mem)
                                mem = x_data_chunks[k] * mask
                                for i in range(len(output)):
                                    mask_i = torch.relu(output[i])
                                    output[i] = x_data_chunks[k] * mask_i
                            mem_chunks.append(mem.clone().detach())

                        else:
                            # Reconstruction test
                            if self.representation_name == RepresentationName.stft:
                                x_data_phase_chunks[k] = y_data_phase_chunks[k].clone()
                            mem, output = y_data_chunks[k].clone(), []
                            mem_chunks.append(mem.clone().detach())

                        # Compute the loss
                        loss = self.loss_fn(mem.view(y_data_chunks[k].shape), y_data_chunks[k],
                                            x_data_phase_chunks[k], y_data_phase_chunks[k])
                        for j in range(len(output)):
                            loss = loss + self.loss_fn(output[j], y_data_chunks[k],
                                                       x_data_phase_chunks[k], y_data_phase_chunks[k])

                        batch_loss_mean += float(loss)

                    if k < self.nb_chunks - 1:
                        self._init_rec()

                # Add the current loss
                batch_loss_mean /= self.nb_chunks
                test_loss_mean += batch_loss_mean

                # Update progress bar
                progress_bar.set_postfix_str(f'Testing Loss: {batch_loss_mean:.5f}')
                progress_bar.update()

                # nb_batches = len(self.dataloader_manager)
                index_list = [0]
                if self.experiment and self.save_mem and batch_index in index_list:
                    self._visualize_output(torch.cat(mem_chunks, -1), torch.cat(x_data_chunks, -1),
                                           torch.cat(y_data_chunks, -1), output, None)
                    self._save_mem(False)

                mem = self._chunks_to_data(mem_chunks)
                x_data_phase = self._chunks_to_data(x_data_phase_chunks) if self.representation_name == RepresentationName.stft else None

                if self.transform_manager:
                    mem = self.transform_manager(mem, mode='inverse_transform')

                mem = self._inverse_preprocess_data(mem, x_data_phase)

                # Save model output to disk
                enhanced_coefficients_dir = os.path.join(tensor_enhanced_coefficients_dir, coefficients_filename)
                self.save_coefficients(mem, index, enhanced_coefficients_dir)

                self._init_rec()

            if self.empty_cache:
                self._empty_cache()

        # Compute mean loss over all batches
        test_loss_mean /= len(self.dataloader_manager)

        # Compute global loss (average loss values across all processes)
        if self.all_reduce_flag:
            test_loss_mean = self._all_reduce_fn(test_loss_mean)

        # Print test mean loss
        print('- Test loss = %.5f' % test_loss_mean)

        # Save mean loss and mean accuracy in the object
        self.results['Test loss'].append(np.round(test_loss_mean, self.nb_digits))

        # Close the tqdm bar
        progress_bar.close()

    def _init_state(self, batch_size: int) -> None:
        """Method that initializes model (SNN) layers state.

        Parameters
        ----------
        batch_size: int
            Data batch size.
        """
        if self.snn_flag:
            if self.use_ddp:
                self.model.module.init_state(batch_size)
            else:
                self.model.init_state(batch_size)

    def _init_rec(self) -> None:
        """Method that initializes model layers records.
        """
        if self.use_ddp:
            self.model.module.init_rec()
        else:
            self.model.init_rec()

    def save_coefficients(self, mem: torch.Tensor, index: torch.Tensor, enhanced_coefficients_dir: str) -> None:
        """Method that saves output encoded data.

        Parameters
        ----------
        mem: torch.Tensor
            Enhanced encoded data.
        index: torch.Tensor
            File name index tensor.
        enhanced_coefficients_dir: str
            Enhanced encoded data directory.
        """
        for i in range(mem.shape[0]):

            index_i = index[i].item()
            str_index_i = str(index_i)
            enhanced_coefficients_i = mem.data[i].clone().detach()

            torch.save(enhanced_coefficients_i, f'{enhanced_coefficients_dir}{str_index_i}.pt')

    def _preprocess_data(self, x_data: torch.Tensor, y_data: torch.Tensor)-> \
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Method that applies preprocessing operations.

        Parameters
        ----------
        x_data: torch.Tensor
            Input encoded data.
        y_data: torch.Tensor
            Target encoded data.
        """
        if self.representation_name == RepresentationName.lca:
            return x_data, y_data, None, None

        elif self.representation_name == RepresentationName.stft:
            x_data_magnitude = torch.abs(x_data)[:, :, :-1, :]
            y_data_magnitude = torch.abs(y_data)[:, :, :-1, :]

            x_data_phase = torch.angle(x_data)[:, :, :-1, :]
            y_data_phase = torch.angle(y_data)[:, :, :-1, :]

            return x_data_magnitude, y_data_magnitude, x_data_phase, y_data_phase

    def _inverse_preprocess_data(self, mem: torch.Tensor, x_data_phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Method that applies inverse preprocessing operations.

        Parameters
        ----------
        mem: torch.Tensor
            Output encoded data.
        x_data_phase: Optional[torch.Tensor]
            Input encoded data phase if encoder is STFT.
        """
        if self.representation_name == RepresentationName.lca:
            return mem
        elif self.representation_name == RepresentationName.stft:
            return pad_coefficients(complex_spectrogram(mem, x_data_phase))

    def _data_to_chunks(self, x_data: torch.Tensor, y_data: torch.Tensor, x_data_phase: Optional[torch.Tensor] = None,
                        y_data_phase: Optional[torch.Tensor] = None) -> \
            Tuple[List[torch.Tensor], List[torch.Tensor], List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
        """Method that splits tensors into a predefined number of chunks.

        Parameters
        ----------
        x_data: torch.Tensor
            Input encoded data.
        y_data: torch.Tensor
            Target encoded data.
        x_data_phase: Optional[torch.Tensor]
            Input encoded data phase if encoder is STFT.
        y_data_phase: Optional[torch.Tensor]
            Target encoded data phase if encoder is STFT.
        """
        if self.nb_chunks == 1:
            x_data_chunks = [x_data]
            y_data_chunks = [y_data]
            x_data_phase_chunks = [x_data_phase]
            y_data_phase_chunks = [y_data_phase]
        else:
            last_time_index = self.nb_steps - (self.nb_steps % self.nb_steps_bin)

            x_data_chunks = x_data[:, :, :, :last_time_index].chunk(self.nb_chunks, 3)
            y_data_chunks = y_data[:, :, :, :last_time_index].chunk(self.nb_chunks, 3)
            if self.representation_name == RepresentationName.stft:
                x_data_phase_chunks = x_data_phase[:, :, :, :last_time_index].chunk(self.nb_chunks, 3)
                y_data_phase_chunks = y_data_phase[:, :, :, :last_time_index].chunk(self.nb_chunks, 3)
            else:
                x_data_phase_chunks = [None] * self.nb_chunks
                y_data_phase_chunks = [None] * self.nb_chunks

        return x_data_chunks, y_data_chunks, x_data_phase_chunks, y_data_phase_chunks

    def _chunks_to_data(self, data_chunks: List[torch.Tensor]) -> torch.Tensor:
        """Method that concatenates tensors.

        Parameters
        ----------
        data_chunks: List[torch.Tensor]
            List of data tensors.
        """
        return torch.cat(data_chunks, -1)

    def prepare_optimizer(self, betas : Tuple[float, float] = (0.9, 0.999), weight_decay: float = 0,
                          amsgrad_flag: bool = False):
        """Method that prepares model optimizer.

        Parameters
        ----------
        betas: Tuple[float, float]
            Coefficients used for computing running averages.
        weight_decay: float
            Weight decay.
        amsgrad_flag: bool
            Boolean that indicates weather to use the AMSGrad variant.
        """

        if self.use_zero:
            self.optimizer = ZeroRedundancyOptimizer(self._get_params(),
                                                     optimizer_class=getattr(optim, self.optimizer_name),
                                                     lr=self.learning_rate
                                                     )
        else:
            if 'Adam' in self.optimizer_name:
                self.optimizer = getattr(optim, self.optimizer_name)(self._get_params(),
                                                                     lr=self.learning_rate,
                                                                     betas=betas,
                                                                     weight_decay=weight_decay,
                                                                     amsgrad=amsgrad_flag,
                                                                     )
            else:
                self.optimizer = getattr(optim, self.optimizer_name)(self._get_params(),
                                                                lr=self.learning_rate,
                                                                )

    def prepare_scheduler(self, lr_scheduler_max_lr: float, lr_scheduler_gamma: float):
        """Method that prepares model scheduler.

        Parameters
        ----------
        lr_scheduler_max_lr: float
            Upper learning rate boundaries in the cycle.
        lr_scheduler_gamma: float
            Multiplicative factor of learning rate decay.
        """

        if self.scheduler_name == 'OneCycleLR':
            steps_per_epoch = len(self.dataloader_manager)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=lr_scheduler_max_lr,
                                                                 steps_per_epoch=steps_per_epoch, epochs=self.nb_epochs,
                                                                 # pct_start=0.1
                                                                 )
        elif self.scheduler_name == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=lr_scheduler_gamma)
        elif self.scheduler_name == 'MultiStepLR':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 20],
                                                                  gamma=lr_scheduler_gamma)
        elif self.scheduler_name == 'ExponentialLR':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_scheduler_gamma)

    def save_checkpoint(self, epoch: int, filepath: Union[str, Path], log_model_flag: bool = False) -> None:
        """Method that saves model state dictionaries in a specified file path.

        Parameters
        ----------
        epoch: int
            Last epoch.
        filepath: Union[str, Path]
            Model state dictionaries file directory.
        log_model_flag: bool
            Boolean that indicates weather to log file to comet ML experiment.
        """

        if self.use_zero:
            self.optimizer.consolidate_state_dict(to=0)

        checkpoint = {'epoch': epoch,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                      'scaler_state_dict': self.scaler.state_dict() if self.scaler is not None else None,
                      }

        # Save checkpoint
        torch.save(checkpoint, filepath)

        if self.empty_cache:
            self._empty_cache()

        if self.experiment and log_model_flag:
            self.experiment.log_model(f'{self.model_name}_pretrained', filepath)

    def load_checkpoint(self, filepath: Union[str, Path]) -> None:
        """Method that loads model state dictionaries from a specified file directory.

        Parameters
        ----------
        filepath: Union[str, Path]
            Model state dictionaries file directory.
        """

        # Load checkpoint
        print('Load model state...')
        # checkpoint = torch.load(filepath, map_location=DataAttributes.device.value)
        # dist.barrier()
        # map_location = f'cuda:{dist.get_rank()}'
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        checkpoint = torch.load(filepath, map_location=map_location)

        # Extract saved training state from non DDP checkpoint to DDP model
        # *****************************************************************
        # state_dict = checkpoint['model_state_dict']
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     k = 'module.' + k
        #     new_state_dict[k] = v
        # self.model.load_state_dict(new_state_dict)
        # ***************************************************************
        # for k, v in checkpoint['model_state_dict'].items():
        #     print('checkpoint k = ', k)
        # ***************************************************************
        # for k, v in self.model.state_dict().items():
        #     print('model k = ', k)
        # ***************************************************************
        #                           ANN to SNN
        # ***************************************************************
        # state_dict = checkpoint['model_state_dict']
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     k = k.replace('ann_layers', 'snn_layers')
        #     new_state_dict[k] = v
        # checkpoint['model_state_dict'] = new_state_dict
        # ***************************************************************

        self.nb_pretrained_epochs = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # if len(self.results['Training loss']) == 0:
        #     self.results['Training loss'].append(np.round(checkpoint['loss'], self.nb_digits))

        if self.empty_cache:
            self._empty_cache()

        # ------------------------------------------------------------------------------
        # print('Model state_dict : ')
        # for param_tensor in self.model.state_dict():
        #     print(param_tensor, '\t', self.model.state_dict()[param_tensor].size())

    def _visualize_output(self, mem: torch.Tensor, x_data: torch.Tensor, y_data: torch.Tensor,
                          output: List[torch.Tensor], epoch: int) -> None:
        """Method that plots model layers output.

        Parameters
        ----------
        mem: torch.Tensor
            Model output.
        x_data: torch.Tensor
            Model input.
        y_data: torch.Tensor
            Model target.
        output: List[torch.Tensor]
            Model intermediate output.
        epoch: int
            Training iteration.
        """
        if self.snn_flag:
            self.visualization_manager.plot_snn_layers(
                spk_rec=self.model.module.spk_rec if self.use_ddp else self.model.spk_rec,
                mem_rec=self.model.module.mem_rec if self.use_ddp else self.model.mem_rec,
                mem=mem,
                x_data=x_data,
                y_data=y_data,
                output=output,
                experiment=self.experiment,
                epoch=epoch)
        else:
            self.visualization_manager.plot_ann_layers(
                out_rec=self.model.module.out_rec if self.use_ddp else self.model.out_rec,
                out=mem,
                x_data=x_data,
                y_data=y_data,
                model_name=self.model_name,
                experiment=self.experiment,
                epoch=epoch)

    def _get_params(self) -> List[torch.Tensor]:
        """Method that creates list of learnable parameters.
        """
        return [params for params in self.model.parameters() if params.requires_grad]

    def _experiment_param_hist(self, epoch: int) -> None:
        """Method that logs model weights to comet ML experiment for histogram visualisation.

        Parameters
        ----------
        epoch: int
            Number of the current training iteration.
        """

        parameters_dict = {'weight': [],
                           'bias': [],
                           'v': [],
                           'alpha': [],
                           'beta': [],
                           'membrane_threshold': [],
                           'scale_factor': [],
                           'bn_weight': [],
                           'bn_bias': [],
                           }

        for index, (name, param) in enumerate(self.model.named_parameters()):

            if 'conv2d_layer.weight' in name:
                parameters_dict['weight'].append(param.clone().detach().cpu().numpy())
            elif 'conv2d_layer.bias' in name:
                parameters_dict['bias'].append(param.clone().detach().cpu().numpy())
            elif 'recurrent2d_layer.v' in name:
                parameters_dict['v'].append(param.clone().detach().cpu().numpy())
            elif 'alpha' in name:
                parameters_dict['alpha'].append(param.clone().detach().cpu().numpy())
            elif 'beta' in name:
                parameters_dict['beta'].append(param.clone().detach().cpu().numpy())
            elif 'membrane_threshold' in name:
                parameters_dict['membrane_threshold'].append(param.clone().detach().cpu().numpy())
            elif 'scale_factor' in name:
                parameters_dict['scale_factor'].append(param.clone().detach().cpu().numpy())
            elif 'bn_layer.weight' in name:
                parameters_dict['bn_weight'].append(param.clone().detach().cpu().numpy())
            elif 'bn_layer.bias' in name:
                parameters_dict['bn_bias'].append(param.clone().detach().cpu().numpy())

        # Log model parameters
        if self.use_ddp:
            layers_name = self.model.module.layers_name
        else:
            layers_name = self.model.layers_name

        v_idx = 0
        mem_thr_idx = 0
        bn_weight_idx = 0
        for k in range(len(parameters_dict['weight'])):

            layer_name = f'{format(k, "02d")}_{layers_name[k]}_'

            self.experiment.log_histogram_3d(parameters_dict['weight'][k], name=f'{layer_name}weight', step=epoch)
            if parameters_dict['bias']:
                self.experiment.log_histogram_3d(parameters_dict['bias'][k], name=f'{layer_name}bias', step=epoch)

            if parameters_dict['v'] and 'Spiking' in layer_name:
                self.experiment.log_histogram_3d(parameters_dict['v'][v_idx], name=f'{layer_name}v', step=epoch)
                v_idx += 1
            if parameters_dict['alpha'] and 'Spiking' in layer_name:
                self.experiment.log_histogram_3d(parameters_dict['alpha'][k], name=f'{layer_name}alpha', step=epoch)
            if parameters_dict['beta'] and 'Spiking' in layer_name:
                self.experiment.log_histogram_3d(parameters_dict['beta'][k], name=f'{layer_name}beta', step=epoch)
            if parameters_dict['membrane_threshold'] and 'Spiking' in layer_name:
                self.experiment.log_histogram_3d(parameters_dict['membrane_threshold'][mem_thr_idx],
                                                 name=f'{layer_name}membrane_threshold', step=epoch)
                mem_thr_idx += 1

            if parameters_dict['scale_factor']:
                self.experiment.log_histogram_3d(parameters_dict['scale_factor'][k],
                                                 name=f'{layer_name}scale_factor', step=epoch)

            if parameters_dict['bn_weight'] and k != 0:
                self.experiment.log_histogram_3d(parameters_dict['bn_weight'][bn_weight_idx],
                                                 name=f'{layer_name}bn_weight', step=epoch)
                if parameters_dict['bn_bias'] and k != 0:
                    self.experiment.log_histogram_3d(parameters_dict['bn_bias'][bn_weight_idx],
                                                     name=f'{layer_name}bn_bias', step=epoch)
                bn_weight_idx += 1

    def _log_grad_norm(self, grad_norm_dict: Optional[dict] = None) -> dict:
        """Method that logs model gradients to comet ML experiment for visualisation.

        Parameters
        ----------
        grad_norm_dict: dict
            Model gradients.
        """
        if grad_norm_dict is None:
            grad_norm_dict = {}
            for index, (name, param) in enumerate(self.model.named_parameters()):
                grad_norm_dict[name] = []
        else:
            for index, (name, param) in enumerate(self.model.named_parameters()):
                if param.grad is not None:
                    norm_grad = param.grad.clone().detach().cpu().norm()
                    grad_norm_dict[name].append(norm_grad)
        return grad_norm_dict

    @staticmethod
    def _empty_cache() -> None:
        """Method that empties cache and runs the garbage collector.
        """
        torch.cuda.empty_cache()
        # Garbage collection
        gc.collect()

    def _find_unused_params(self) -> None:
        """Method that prints model unused parameters.
        """
        # Find unused parameters
        print('- UNUSED PARAMS: ')
        print('---------------')
        for name, param in self.model.named_parameters():
            if param.grad is None:
                print(f'\t - param {name} is not used during training.')

    def _clamp_neuron_params(self) -> None:
        """Method that clamps model (SNN) neuron parameters.
        """
        if self.use_ddp:
            if self.model.module.train_neuron_parameters:
                self.model.module.clamp_neuron_parameters()
        else:
            if self.model.train_neuron_parameters:
                self.model.clamp_neuron_parameters()

    def _clip_grad(self) -> None:
        """Method that clips model gradients.
        """
        if self.clip_type == 'value':
            # Gradient Value Clipping
            # clip_value = 0.5
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_value)
        elif self.clip_type == 'norm':
            # Gradient Norm Clipping
            # max_norm = 0.001
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_value, norm_type=2)

    def _save_mem(self, value: bool) -> None:
        """Method that updates model `save_mem` attribute.

        Parameters
        ----------
        value: bool
            Boolean that indicates weather to track output.
        """
        if self.use_ddp:
            self.model.module.save_mem = value
        else:
            self.model.save_mem = value

    def _all_reduce_fn(self, metric_mean: float) -> float:
        """Method that reduces data across DDP processors.

        Parameters
        ----------
        metric_mean: float
            Value to be reduced.
        """

        # Compute global metric (average loss values across all processes)
        metric_mean_tensor = torch.tensor([metric_mean], dtype=self.dtype).to(self.device)
        dist.all_reduce(metric_mean_tensor)
        metric_mean = metric_mean_tensor.item() / self.world_size
        return metric_mean

