"""
File:
    model/SpeechEnhancer.py

Description:
    Defines the SpeechEnhancer class.
"""

# Import libraries
import gc
from comet_ml import Experiment
import os
from typing import Optional
from torchinfo import summary
from datetime import timedelta
from datetime import datetime
import numpy as np
import math
import random
import json
import torch
import torch.quantization
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
from src.model.utils import random_index
from src.data.DatasetManager import DatasetManager
from src.model.SurrogateGradient import SuperSpike, SigmoidDerivative, ATan, PiecewiseLinear, SpikeFunc
from src.model.SpikingModel import FCSNN, CSNN, UNetSNN, ResBottleneckUNetSNN
from src.model.ArtificialModel import CNN, UNet, ResBottleneckUNet
from src.model.TrainValidTestManager import TrainValidTestManager
from src.evaluation.EvaluationManager import EvaluationManager
from src.visualization.VisualizationManager import VisualizationManager
from src.data.constants import DataDirectories, AudioParameters, RepresentationName
from src.model.constants import SNNstr, ANNstr


class SpeechEnhancer:
    """Class that implements the speech enhancer.
    """

    def __init__(self,
                 model_name: str, use_mask: bool, representation_name: str, representation_dir_name: str,
                 transform_name: Optional[list[str]], reconstruct_flag: bool, compute_representation: bool,
                 batch_size: int, k: float, tau: float, tau_syn: float, tau_mem: float,
                 tau_syn_out: float, tau_mem_out: float, time_step: float, membrane_threshold: float, decay_input: bool,
                 spiking_mode: str, reset_mode: str, detach_reset: bool, weight_mean: float, weight_std: float,
                 weight_gain: float, weight_init_dist: str, input_dim: int, hidden_dim_list: list,
                 hidden_channels_list: list, kernel_size: tuple, stride: tuple, padding: Optional[tuple],
                 dilation: tuple, bias: bool, padding_mode: str, pooling_flag: bool, pooling_type: str,
                 use_same_layer: bool, recurrent_flag: bool, neuron_model: str, train_neuron_parameters: bool,
                 neuron_parameters_init_dist: str, upsample_mode: str, scale_flag: bool, scale_factor: float,
                 bn_flag: bool, dropout_flag: bool, dropout_p: float, skip_connection_type: str,
                 nb_residual_block: int, residual_skip_connection_type: str, use_intermediate_output: bool,
                 loss_name: list, loss_weight: list, loss_bias: list, surrogate_name: str, surrogate_scale: Optional[float],
                 activation_fn: list[str], optimizer_name: str, learning_rate: float, betas: tuple,
                 scheduler_name: Optional[str], lr_scheduler_max_lr: float, lr_scheduler_gamma: float,
                 clip_type: Optional[str], clip_value: float, nb_epochs: int, nb_warmup_epochs: int,
                 nb_steps_bin: Optional[int], truncated_bptt_ratio: Optional[int],
                 pretrained_flag: bool, model_file_name_id: Optional[int], save_mem: bool, save_model: bool,
                 perceptual_metric_flag: bool, train_flag: bool, hpc_flag: bool, use_ddp: bool, dist_backend: str,
                 evaluate_flag: bool, use_amp: bool, use_zero: bool, pin_memory: bool, num_workers: int,
                 prefetch_factor: int, deterministic: bool, seed: int, empty_cache: bool, debug_flag: bool,
                 comet_ml_params: Optional[dict],
                 ) -> None:

        self._set_seed(seed=int(seed), deterministic=deterministic)
        if deterministic:
            self.seed_worker = self._set_seed_worker
        else:
            self.seed_worker = None
        self._set_debug_apis()

        # -----------------------------------------     Set CometML config     -----------------------------------------

        print('Preparing CometML experiment instance...')

        debug_str = ''
        self.debug_flag = debug_flag
        if debug_flag:
            debug_str = '__DEBUG__'

        # Define experiment instance
        comet_ml_params_flag = True
        for key, value in comet_ml_params.items():
            if value is None:
                comet_ml_params_flag = False

        self.experiment = None
        if comet_ml_params_flag:
            self.experiment = Experiment(api_key=comet_ml_params['api_key'], workspace=comet_ml_params['workspace'],
                                         project_name=f'{comet_ml_params["project_name"]}{debug_str}',
                                         log_code=False, log_graph=False, auto_param_logging=False,
                                         auto_metric_logging=False, auto_output_logging='simple',
                                         log_git_metadata=False, log_git_patch=False, log_env_details=True,
                                         log_env_gpu=True, log_env_cpu=True, log_env_host=True,
                                         # display_summary=None,
                                         )

        if reconstruct_flag:
            task = 'Reconstruction'
        else:
            task = 'Enhancement'

        task_name = f'Speech{task}'

        self.train_flag = train_flag

        if self.train_flag:
            exp_type = 'Train'
        else:
            exp_type = 'Test'

        cmt_exp_name = f'{task_name}_{exp_type}_{model_name}'

        if self.experiment:
            self.experiment.set_name(name=debug_str+cmt_exp_name)

        # -----------------------------------------       Set DDP config       -----------------------------------------
        if use_ddp:
            print('Preparing DistributedDataParallel process group...')

        self.hpc_flag = hpc_flag
        self.use_ddp = use_ddp
        self.evaluate_flag = evaluate_flag
        self.use_amp = use_amp
        self.use_zero = use_zero
        self.empty_cache = empty_cache
        valid_dist_backend = ['nccl', 'gloo']
        assert dist_backend in valid_dist_backend, \
            f'dist_backend parameter "{dist_backend}" should be in {valid_dist_backend}'
        if hpc_flag:
            self.dist_backend = dist_backend
        else:
            self.dist_backend = 'gloo'
        self.init_method = None
        self.pin_memory = pin_memory
        self.num_workers = int(num_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.seed = seed
        self.drop_last = True
        if self.use_ddp and self.hpc_flag:
            self.world_size = int(os.environ.get('SLURM_NTASKS'))   # torch.cuda.device_count()
        else:
            self.world_size = 1
        self.current_device = 0
        self.rank = 0
        if self.use_ddp:
            self.setup()
            print(f'Running DistributedDataParallel on rank {self.rank}...')

        # -----------------------------------------      Set data config       -----------------------------------------

        self.dtype = torch.float                                        # torch.half
        # self.memory_format = torch.channels_last                      # torch.channels_last, torch.contiguous_format

        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.rank}')
            print('GPU detected...')
            # from torch.utils.collect_env import get_pretty_env_info
            # env_str = get_pretty_env_info()
        else:
            self.device = torch.device('cpu')
            print('No GPU. switching to CPU...')

        # -----------------------------------------      Set directories       -----------------------------------------

        print('From Rank: {}, ==> Setting data directories...'.format(self.rank))

        # Set data attribute
        valid_representation_name = [RepresentationName.lca, RepresentationName.stft]
        assert representation_name in valid_representation_name, \
            f'representation_name parameter "{representation_name}" should be in {valid_representation_name}'

        lca_dir = os.path.join(os.getcwd(), 'src', 'lca')
        if representation_name == RepresentationName.lca and not os.path.exists(lca_dir):
            representation_name = RepresentationName.stft
            print(f"Warning: {RepresentationName.lca} representation module does not exist. "
                  f"Defaulting to {RepresentationName.stft} representation.")

        self.representation_name = representation_name
        self.representation_dir_name = representation_dir_name
        self.transform_name = transform_name
        self.compute_representation = compute_representation
        self.reconstruct_flag = reconstruct_flag
        self.task_name = task_name

        # Set data directory
        if self.hpc_flag and not compute_representation and not debug_flag:
            data_files_dir = os.path.join(str(os.environ.get('SLURM_TMPDIR')), 'data',
                                          f'{task_name}_{DataDirectories.data_dirname}')
        else:
            data_files_dir = os.path.join(DataDirectories.project_dir,
                                          f'{task_name}_{DataDirectories.data_dirname}')

        # Set pytorch model directory
        model_file_name = f'{task_name}_{model_name}_InpDim={input_dim}'
        if model_name == SNNstr.FCSNNstr:
            assert hidden_dim_list is not None, f'hidden_dim_list parameter "{hidden_dim_list}" should be a list of integers'
            str_hidden_dim_list = ''.join([str(item) + '-' for item in hidden_dim_list])[:-1]
            model_file_name += f'_HddnLyrs={str_hidden_dim_list}'
        elif hidden_channels_list is not None:
            # Set padding parameter
            if padding is None:
                # Padding: 'Same'
                padding_0 = int(np.ceil(((dilation[0] * (kernel_size[0] - 1)) + 1 - stride[0]) / 2))
                padding_1 = int(np.ceil(((dilation[1] * (kernel_size[1] - 1)) + 1 - stride[1]) / 2))
                padding = (padding_0, padding_1)

            str_hidden_dim_list = ''.join([str(item) + '_' for item in hidden_channels_list])[:-1]
            model_file_name += f'_HddnChnnls={str_hidden_dim_list}' \
                               f'_KrnlSz={int(kernel_size[0])}_{int(kernel_size[1])}' \
                               f'_Strd={int(stride[0])}_{int(stride[1])}'

        if model_name in [SNNstr.ResBottleneckUNetSNNstr]:
            model_file_name += f'_nbRB={int(nb_residual_block)}'

        if use_amp:
            model_file_name += '_amp'

        if model_file_name_id is not None:
            model_file_name += f'_id={model_file_name_id}'

        trained_models_dir = os.path.join(DataDirectories.project_dir, DataDirectories.trained_models_dirname)
        self.model_file_dir = os.path.join(trained_models_dir, model_file_name + '.pt')

        # Set experiment directory
        experiments_dir = os.path.join(DataDirectories.project_dir, DataDirectories.experiments_dirname)
        if self.rank == 0:
            self._check_dir(dir_=experiments_dir)
        if self.use_ddp:
            dist.barrier()

        if self.hpc_flag:
            experiment_id = int(os.environ.get('SLURM_JOB_ID'))
        else:
            # experiment_id = random.randint(0, 100000)
            experiment_id = ''

        if self.use_ddp and self.world_size > 1:
            experiment_id_tensor = torch.tensor([experiment_id], dtype=torch.int).to(self.device)
            dist.broadcast(experiment_id_tensor, src=0)

        experiment_file_name = f'{debug_str}{task_name}_{exp_type}_{model_name}_' \
                               f'{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}_' \
                               f'{experiment_id}'

        self.experiment_files_dir = os.path.join(experiments_dir, experiment_file_name)
        self.plots_dir = os.path.join(self.experiment_files_dir, 'plots')

        if self.rank == 0:
            os.makedirs(self.experiment_files_dir)
            os.makedirs(self.plots_dir)
        if self.use_ddp:
            dist.barrier()

        print(f'\t - experiment_file_name : {experiment_file_name}')
        print(f'\t - model_file_name : {model_file_name}')

        if self.experiment:
            self.experiment.log_other('experiment_file_name', experiment_file_name)
            self.experiment.log_other('model_file_name', model_file_name)
            self.experiment.log_other('rank', self.rank)
            self.experiment.log_other('rank_id', os.environ.get('MASTER_ADDR'))
            self.experiment.log_other('empty_cache', self.empty_cache)
            self.experiment.log_other('num_workers', self.num_workers)
            if hpc_flag:
                self.experiment.log_other('job_id', os.environ.get('SLURM_JOB_ID'))

        # model_checkpoint_dir = os.path.join(self.experiment_files_dir, 'checkpoint')
        # if self.rank == 0:
        #     self._check_dir(dir_=model_checkpoint_dir)
        # if self.use_ddp:
        #     dist.barrier()

        model_checkpoint_dir = self.experiment_files_dir

        self.exp_model_file_dir = os.path.join(model_checkpoint_dir, f'{model_file_name}.pt')

        # -----------------------------------------   Set dataset instances    -----------------------------------------

        print('From Rank: {}, ==> Preparing dataset managers...'.format(self.rank))

        # Define visualization manager
        self.visualization_manager = VisualizationManager()

        valid_upsample_mode = ['nearest', 'bilinear']
        assert upsample_mode in valid_upsample_mode, \
            f'upsample_mode parameter "{upsample_mode}" should be in {valid_upsample_mode}'
        self.upsample_mode = upsample_mode
        self.scale_flag = scale_flag
        self.scale_factor = scale_factor

        self.bn_flag = bn_flag
        self.dropout_flag = dropout_flag
        self.dropout_p = dropout_p

        valid_skip_connection_type = ['cat_', 'add_']
        assert skip_connection_type in valid_skip_connection_type, \
            f'skip_connection_type parameter "{skip_connection_type}" should be in {valid_skip_connection_type}'
        self.skip_connection_type = skip_connection_type

        self.nb_residual_block = int(nb_residual_block)
        valid_residual_skip_connection_type = ['add_', 'and_', 'iand_']
        assert residual_skip_connection_type in valid_residual_skip_connection_type, \
        f'residual_skip_connection_type parameter "{residual_skip_connection_type}" should be in ' \
        f'{valid_residual_skip_connection_type}'
        self.residual_skip_connection_type = residual_skip_connection_type

        self.use_intermediate_output = use_intermediate_output

        valid_loss_name = ['mse_loss', 'l1_loss', 'lsd_loss', 'time_mse_loss', 'huber_loss', 'stoi_loss', 'si_snr_loss',
                           'si_sdr_loss']
        for loss_name_i in loss_name:
            assert loss_name_i in valid_loss_name, f'loss_name parameter "{loss_name_i}" should be in {valid_loss_name}'
        self.loss_name = loss_name
        self.loss_weight = loss_weight
        self.loss_bias = loss_bias

        self.nb_epochs = int(nb_epochs)
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.betas = betas

        valid_clip_type = [None, 'value', 'norm']
        assert clip_type in valid_clip_type, f'clip_type parameter "{clip_type}" should be in {valid_clip_type}'

        self.clip_type = clip_type
        self.clip_value = clip_value

        valid_scheduler_name = [None, 'OneCycleLR', 'StepLR', 'MultiStepLR', 'ExponentialLR']
        assert scheduler_name in valid_scheduler_name, \
            f'scheduler_name parameter "{scheduler_name}" should be in {valid_scheduler_name}'
        self.scheduler_name = scheduler_name
        self.lr_scheduler_max_lr = lr_scheduler_max_lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        if scheduler_name == 'OneCycleLR_':
            self.nb_warmup_epochs = 0
        else:
            self.nb_warmup_epochs = int(nb_warmup_epochs)
        self.pretrained_flag = pretrained_flag
        self.save_mem = save_mem
        self.save_model = save_model

        # Set spiking mode parameter
        valid_spiking_mode = ['binary', 'graded']
        assert spiking_mode in valid_spiking_mode, \
            f'spiking_mode parameter "{spiking_mode}" should be in {valid_spiking_mode}'

        # Overwrite naive spike function by the spike_fn nonlinearity
        if surrogate_name == 'SuperSpike':
            self.spike_fn = SuperSpike
        elif surrogate_name == 'SigmoidDerivative':
            self.spike_fn = SigmoidDerivative
        elif surrogate_name == 'ATan':
            self.spike_fn = ATan
        elif surrogate_name == 'PiecewiseLinear':
            self.spike_fn = PiecewiseLinear
        elif surrogate_name == 'SpikeFunc':
            self.spike_fn = SpikeFunc
        else:
            valid_surrogate_name = ['SuperSpike', 'SigmoidDerivative', 'ATan', 'PiecewiseLinear', 'SpikeFunc']
            assert surrogate_name in valid_surrogate_name, \
                f'surrogate_name parameter "{surrogate_name}" should be in {valid_surrogate_name}'

        self.surrogate_name = surrogate_name
        self.spike_fn.spiking_mode = spiking_mode
        self.spiking_mode = self.spike_fn.spiking_mode

        if surrogate_scale is not None:
            self.spike_fn.surrogate_scale = surrogate_scale
        self.surrogate_scale = self.spike_fn.surrogate_scale

        valid_activation_fn = [None, 'sigmoid', 'relu', 'lrelu', 'prelu', 'tanh']
        for i in range(len(activation_fn)):
            assert activation_fn[i] in valid_activation_fn, \
                f'activation_fn[{i}] parameter "{activation_fn[i]}" should be in {valid_activation_fn}'

        self.activation_fn = activation_fn

        # Set perceptual_metric_flag attribute
        self.perceptual_metric_flag = perceptual_metric_flag

        # Set batch size
        self.batch_size = int(batch_size)
        self.batch_size_per_rank = max(self.batch_size // self.world_size, 1)
        self.batch_size_per_rank_eval = 8

        if debug_flag:
            self.batch_size_per_rank_eval = self.batch_size_per_rank

        # Set input dim
        self.input_dim = input_dim
        self.output_dim = self.input_dim

        # Define dataset managers
        self.dataset_manager_train = DatasetManager(data_files_dir=data_files_dir,
                                                    data_load=DataDirectories.data_load_train,
                                                    experiment_files_dir=self.experiment_files_dir,
                                                    plots_dir=self.plots_dir, dtype=self.dtype,
                                                    representation_name=representation_name,
                                                    representation_dir_name=representation_dir_name,
                                                    transform_name=self.transform_name,
                                                    compute_representation=compute_representation,
                                                    reconstruct_flag=reconstruct_flag, use_ddp=use_ddp,
                                                    experiment=self.experiment, debug_flag=debug_flag)
        self.dataset_manager_valid = DatasetManager(data_files_dir=data_files_dir,
                                                    data_load=DataDirectories.data_load_valid,
                                                    experiment_files_dir=self.experiment_files_dir,
                                                    plots_dir=self.plots_dir, dtype=self.dtype,
                                                    representation_name=representation_name,
                                                    representation_dir_name=representation_dir_name,
                                                    transform_name=self.transform_name,
                                                    compute_representation=compute_representation,
                                                    reconstruct_flag=reconstruct_flag, use_ddp=use_ddp,
                                                    experiment=self.experiment, debug_flag=debug_flag)
        self.dataset_manager_test = DatasetManager(data_files_dir=data_files_dir,
                                                   data_load=DataDirectories.data_load_test,
                                                   experiment_files_dir=self.experiment_files_dir,
                                                   plots_dir=self.plots_dir, dtype=self.dtype,
                                                   representation_name=representation_name,
                                                   representation_dir_name=representation_dir_name,
                                                   transform_name=self.transform_name,
                                                   compute_representation=compute_representation,
                                                   reconstruct_flag=reconstruct_flag, use_ddp=use_ddp,
                                                   experiment=self.experiment, debug_flag=debug_flag)

        # Define dataloader managers
        self.dataloader_manager_train = None
        self.dataloader_manager_valid = None
        self.dataloader_manager_test = None

        self.nb_steps_bin = nb_steps_bin
        if nb_steps_bin is None:
            self.nb_steps_bin = self.dataset_manager_train.nb_steps

        if truncated_bptt_ratio:
            self.truncated_bptt_ratio = truncated_bptt_ratio
        else:
            self.truncated_bptt_ratio = 1

        # -----------------------------------------         SNN model          -----------------------------------------

        print('From Rank: {}, ==> Setting model hyperparameters...'.format(self.rank))

        # Set predefined LIF parameters
        self.time_step = time_step
        self.tau_syn = tau_syn
        self.tau_mem = tau_mem
        self.tau_syn_out = tau_syn_out
        self.tau_mem_out = tau_mem_out

        # Set predefined PLIF parameters
        self.tau = tau

        # Set predefined KLIF parameters
        self.k = k

        valid_neuron_parameters_init_dist = ['constant_', 'normal_', 'uniform_']
        assert neuron_parameters_init_dist in valid_neuron_parameters_init_dist, \
            f'neuron_parameters_init_dist parameter "{neuron_parameters_init_dist}" ' \
            f'should be in {valid_neuron_parameters_init_dist}'

        valid_reset_mode = ['hard_reset', 'soft_reset']
        assert reset_mode in valid_reset_mode, \
            f'reset_mode parameter "{reset_mode}" should be in {valid_reset_mode}'

        # Compute neuron_parameters
        self.neuron_model = neuron_model
        self.neuron_parameters = {'train_neuron_parameters': train_neuron_parameters,
                                  'neuron_parameters_init_dist': neuron_parameters_init_dist,
                                  'decay_input': decay_input,
                                  'recurrent_flag': recurrent_flag,
                                  'reset_mode': reset_mode,
                                  'detach_reset': detach_reset,
                                  'membrane_threshold': membrane_threshold,
                                  }
        if neuron_model == 'lif' or neuron_model == 'klif':
            self.neuron_parameters['alpha'] = float(np.exp(-time_step / tau_syn))
            self.neuron_parameters['beta'] = float(np.exp(-time_step / tau_mem))
            self.neuron_parameters['alpha_out'] = float(np.exp(-time_step / tau_syn_out))
            self.neuron_parameters['beta_out'] = float(np.exp(-time_step / tau_mem_out))
            if neuron_model == 'klif':
                self.neuron_parameters['k'] = k
        elif neuron_model == 'plif':
            self.neuron_parameters['alpha'] = - math.log(tau - 1)
            self.neuron_parameters['beta'] = - math.log(tau - 1)
            self.neuron_parameters['alpha_out'] = - math.log(tau - 1)
            self.neuron_parameters['beta_out'] = - math.log(tau - 1)
        elif neuron_model == 'if':
            pass
        else:
            valid_neuron_model = ['lif', 'plif', 'klif', 'if']
            assert neuron_model in valid_neuron_model, \
                f'neuron_model parameter "{neuron_model}" should be in {valid_neuron_model}'

        if 'SNN' in model_name:
            print(f'\t - neuron_model = {self.neuron_model}')
            print(f'\t - neuron_parameters : ')
            for k, v in self.neuron_parameters.items():
                print(f'\t\t - {k} : {v}')

        # FCSNN
        self.hidden_dim_list = hidden_dim_list

        # CSNN, SESNN, ADDSESNN
        self.hidden_channels_list = hidden_channels_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        valid_padding_mode = ['zeros', 'reflect', 'replicate', 'circular']
        assert padding_mode in valid_padding_mode, \
            f'padding_mode parameter "{padding_mode}" should be in {valid_padding_mode}'

        self.padding_mode = padding_mode

        self.pooling_flag = pooling_flag

        valid_pooling_type = ['max', 'avg']
        assert pooling_type in valid_pooling_type, \
            f'pooling_type parameter "{pooling_type}" should be in {valid_pooling_type}'

        self.pooling_type = pooling_type

        self.use_same_layer = use_same_layer

        self.model = None

        valid_model_name = [SNNstr.FCSNNstr, SNNstr.CSNNstr, SNNstr.UNetSNNstr, SNNstr.ResBottleneckUNetSNNstr,
                            ANNstr.CNNstr, ANNstr.UNetstr, ANNstr.ResBottleneckUNetstr]
        assert model_name in valid_model_name, \
            f'model_name parameter "{model_name}" should be in {valid_model_name}'
        self.model_name = model_name
        self.use_mask = use_mask

        valid_weight_init_dist = ['normal_', 'uniform_', 'kaiming_normal_', 'kaiming_uniform_', 'xavier_uniform_']
        assert weight_init_dist in valid_weight_init_dist, \
            f'weight_init_dist parameter "{weight_init_dist}" should be in {valid_weight_init_dist}'

        self.weight_init = {'weight_mean': weight_mean,
                            'weight_std': weight_std,
                            'weight_gain': weight_gain,
                            'weight_init_dist': weight_init_dist}

        # -----------------------------------------   SNN train/test manager   -----------------------------------------

        print('From Rank: {}, ==> Preparing Training/Testing managers...'.format(self.rank))

        # Add model graph to experiment instance
        if self.experiment:
            self.experiment.set_model_graph(str(self.model), overwrite=True)

        self.model_train_manager = None
        self.model_test_manager = None

        # -----------------------------------------    hyperparameters_dict    -----------------------------------------

        # Save SNN hyperparameters to json file
        hyperparameters_dc = self.get_params_dict()

        if self.rank == 0:
            # Save parameters to json file
            self.save_json_file(file_name='params', file_rec_dict={'hyperparameters': hyperparameters_dc})

        # Log hyperparameters
        if self.experiment:
            self.experiment.log_parameters(hyperparameters_dc)

    @staticmethod
    def _check_dir(dir_: str) -> None:
        """Method that check if a particular directory exist or not.

        Parameters
        ----------
        dir_: str
            Directory to be checked before creating.
        """
        if not os.path.exists(dir_):
            os.makedirs(dir_)

    def setup(self, verbose: bool = True) -> None:
        """Method that defines DDP instance.

        Parameters
        ----------
        verbose: bool
            Boolean that indicates weather to print specific output.
        """
        if self.hpc_flag:
            # self.num_workers = 2
            # self.num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK'))
            # self.prefetch_factor = 4
            # port = random.randint(49152, 65535)
            ngpus_per_node = int(torch.cuda.device_count())
            # self.init_method = 'tcp://{}:{}'.format(str(os.environ.get('MASTER_ADDR')), 3456)
            self.init_method = 'tcp://127.0.0.1:3456'
            local_rank = int(os.environ.get('SLURM_LOCALID'))
            self.rank = int(os.environ.get('SLURM_NODEID')) * ngpus_per_node + local_rank

            self.current_device = local_rank
            torch.cuda.set_device(self.current_device)

            # Initialize the process group
            dist.init_process_group(backend=self.dist_backend,
                                    init_method=self.init_method,
                                    world_size=self.world_size,
                                    rank=self.rank,
                                    timeout=timedelta(seconds=180000)                       # timeout = 50 hours
                                    )
        else:
            # 'tcp://127.0.0.1:3456'
            os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
            os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', str(12355))

            self.init_method = f'{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}'

            torch.cuda.set_device(self.current_device)

            # Initialize the process group
            dist.init_process_group(backend=self.dist_backend,
                                    rank=self.rank,
                                    world_size=self.world_size)

        if verbose:
            print('\t - pin_memory = ', self.pin_memory)
            print('\t - num_workers = ', self.num_workers)
            print('\t - prefetch_factor = ', self.prefetch_factor)
            print('\t - init_method = ', self.init_method)
            print('\t - dist_backend = ', self.dist_backend)
            print('\t - world_size = ', self.world_size)
            print('\t - current_device = ', self.current_device)
            print('\t - rank = ', self.rank)

    @staticmethod
    def _cleanup() -> None:
        """Method that destroys DDP process group.
        """
        dist.destroy_process_group()

    def plot_input_data(self, dataset_manager: DatasetManager, show_flag: bool = False, verbose: bool = False) -> None:
        """Method that plots signal data for multiple signals.

        Parameters
        ----------
        dataset_manager: DatasetManager
            Dataset manager instance.
        show_flag: bool
            Boolean that indicates weather to display plots.
        verbose: bool
            Boolean that indicates weather to print specific output.
        """

        # index = [np.random.randint(0, dataset_manager.cleanspeech_coefficients.shape[0] - 1)]
        # index = [int(10 / dist.get_world_size() + dist.get_rank())]
        index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        for index in index_list:
            # Clean signal
            coefficients = dataset_manager.get_coefficients(dataset_manager.tensor_cleanspeech_coefficients_dir, index)
            self.visualization_manager.plot_signal_data(audio_dir=dataset_manager.cleanspeech_dir, index=index,
                                                        plots_dir=self.plots_dir,
                                                        representation_name=self.representation_name,
                                                        coefficients=coefficients.view(coefficients.shape[-2],
                                                                                       coefficients.shape[-1]
                                                                                       ).detach().cpu().numpy(),
                                                        signal_type='clean idx_' + str(index),
                                                        experiment=self.experiment, show_flag=show_flag, verbose=verbose)
            # Noisy signal
            coefficients = dataset_manager.get_coefficients(dataset_manager.tensor_noisyspeech_coefficients_dir, index)
            self.visualization_manager.plot_signal_data(audio_dir=dataset_manager.noisyspeech_dir, index=index,
                                                        plots_dir=self.plots_dir,
                                                        representation_name=self.representation_name,
                                                        coefficients=coefficients.view(coefficients.shape[-2],
                                                                                       coefficients.shape[-1]
                                                                                       ).detach().cpu().numpy(),
                                                        signal_type='noisy idx_' + str(index),
                                                        experiment=self.experiment, show_flag=show_flag, verbose=verbose)


    def plot_output_data(self, dataset_manager: DatasetManager, show_flag: bool = False) -> None:
        """Method that plots signal data for output (enhanced) test signals.

        Parameters
        ----------
        dataset_manager: DatasetManager
            Dataset manager instance.
        show_flag: bool
            Boolean that indicates weather to display plots.
        """

        # index = np.random.randint(0, dataset_manager.cleanspeech_coefficients.shape[0] - 1)
        # index = int(10 / dist.get_world_size() + dist.get_rank())
        index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        for index in index_list:
            # Enhanced signal
            coefficients = dataset_manager.get_coefficients(self.dataset_manager_test.tensor_enhanced_coefficients_dir,
                                                            index)
            self.visualization_manager.plot_signal_data(audio_dir=self.dataset_manager_test.enhancedspeech_dir,
                                                        index=index, plots_dir=self.plots_dir,
                                                        representation_name=self.representation_name,
                                                        coefficients=coefficients.view(coefficients.shape[-2],
                                                                                       coefficients.shape[-1]
                                                                                       ).detach().cpu().numpy(),
                                                        signal_type='enhanced idx_' + str(index),
                                                        experiment=self.experiment,
                                                        plot_coefficients=True
                                                        if self.representation_name == RepresentationName.stft
                                                        else False,
                                                        show_flag=show_flag)


    def plot_weights(self, show_flag: bool = False) -> None:
        """Method that plots LCA weights.

        Parameters
        ----------
        show_flag: bool
            Boolean that indicates weather to display plots.
        """
        weights_fig_dir = os.path.join(self.plots_dir, 'GT filterbank magnitude in dB')
        self.visualization_manager._plot_kernel(weights=
                                               self.dataset_manager_train.get_weights().detach().cpu().numpy()[:, 0, :],
                                               sample_rate=AudioParameters.sample_rate,
                                               fig_dir=weights_fig_dir,
                                               show_flag=show_flag)

    def get_params_dict(self):
        """Method that saves hyperparameters dictionary.
        """
        hyperparameters_dc = {'representation_name': self.representation_name,
                              'representation_dir_name': self.representation_dir_name,
                              'transform_name': self.transform_name,
                              'compute_representation': self.compute_representation, 'task_name': self.task_name,
                              'weight_init': self.weight_init, 'input_dim': self.input_dim,
                              'output_dim': self.output_dim, 'nb_epochs': self.nb_epochs,
                              'nb_warmup_epochs': self.nb_warmup_epochs, 'nb_steps_bin': self.nb_steps_bin,
                              'batch_size': self.batch_size, 'batch_size_per_rank': self.batch_size_per_rank,
                              'loss_name': self.loss_name, 'loss_weight': self.loss_weight, 'loss_bias': self.loss_bias,
                              'optimizer_name': self.optimizer_name, 'learning_rate': self.learning_rate,
                              'betas': self.betas, 'clip_type': self.clip_type, 'clip_value': self.clip_value,
                              'scheduler_name': self.scheduler_name, 'model_name': self.model_name, 'use_mask': self.use_mask,
                              'scale_flag': self.scale_flag, 'scale_factor': self.scale_factor, 'bn_flag': self.bn_flag,
                              'dropout_flag': self.dropout_flag, 'evaluate_flag': self.evaluate_flag,
                              'use_amp': self.use_amp, 'use_zero': self.use_zero, 'use_ddp': self.use_ddp,
                              'device': str(self.device), 'dtype': str(self.dtype),
                              }
        if 'SNN' in self.model_name:
            hyperparameters_dc['time_step'] = self.time_step
            hyperparameters_dc['neuron_model'] = self.neuron_model
            hyperparameters_dc['spike_fn'] = str(self.spike_fn.__qualname__)
            hyperparameters_dc['surrogate_scale'] = str(self.surrogate_scale)
            hyperparameters_dc['spiking_mode'] = str(self.spiking_mode)
            hyperparameters_dc['truncated_bptt_ratio'] = str(self.truncated_bptt_ratio)
            hyperparameters_dc['use_intermediate_output'] = str(self.use_intermediate_output)

        if self.model_name in [SNNstr.UNetSNNstr, SNNstr.ResBottleneckUNetSNNstr, ANNstr.UNetstr,
                               ANNstr.ResBottleneckUNetstr]:
            hyperparameters_dc['skip_connection_type'] = str(self.skip_connection_type)
            if self.model_name in [SNNstr.ResBottleneckUNetSNNstr, ANNstr.ResBottleneckUNetstr]:
                hyperparameters_dc['residual_skip_connection_type'] = str(self.residual_skip_connection_type)
                hyperparameters_dc['nb_residual_block'] = str(self.nb_residual_block)

        if self.model_name == SNNstr.FCSNNstr and self.hidden_dim_list is not None:
            hyperparameters_dc['hidden_dim_list'] = str(self.hidden_dim_list)
        else:
            hyperparameters_dc['bias'] = str(self.bias)
            hyperparameters_dc['padding_mode'] = str(self.padding_mode)
            hyperparameters_dc['pooling_flag'] = str(self.pooling_flag)
            hyperparameters_dc['pooling_type'] = str(self.pooling_type)
            hyperparameters_dc['use_same_layer'] = str(self.use_same_layer)
            hyperparameters_dc['upsample_mode'] = str(self.upsample_mode)
            if self.hidden_channels_list is not None:
                hyperparameters_dc['hidden_channels_list'] = str(self.hidden_channels_list)
                hyperparameters_dc['kernel_size'] = str(self.kernel_size)
                hyperparameters_dc['stride'] = str(self.stride)
                hyperparameters_dc['padding'] = str(self.padding)
                hyperparameters_dc['dilation'] = str(self.dilation)

        if self.neuron_model == 'lif' or self.neuron_model == 'klif':
            hyperparameters_dc['tau_syn'] = self.tau_syn
            hyperparameters_dc['tau_mem'] = self.tau_mem
            hyperparameters_dc['tau_syn_out'] = self.tau_syn_out
            hyperparameters_dc['tau_mem_out'] = self.tau_mem_out
            if self.neuron_model == 'klif':
                hyperparameters_dc['k'] = self.k
        elif self.neuron_model == 'plif':
            hyperparameters_dc['tau'] = self.tau

        hyperparameters_dc['neuron_parameters'] = self.neuron_parameters

        if self.scheduler_name == 'OneCycleLR_':
            hyperparameters_dc['lr_scheduler_max_lr'] = self.lr_scheduler_max_lr

        elif self.scheduler_name is not None:
            hyperparameters_dc['lr_scheduler_gamma'] = self.lr_scheduler_gamma

        if self.dropout_flag:
            hyperparameters_dc['dropout_p'] = self.dropout_p

        if self.experiment:
            hyperparameters_dc['url'] = self.experiment.url

        return hyperparameters_dc

    def save_json_file(self, file_name: str, file_rec_dict: dict) -> None:
        """Method that saves file_rec_dict within a json file.

        Parameters
        ----------
        file_name: str
            File name.
        file_rec_dict: dict
            Dictionary to  be saved to a file.
        """
        # Dump the experiment configuration dictionary into a json file
        experiment_rec_dir = os.path.join(self.experiment_files_dir, f'{file_name}.json')
        experiment_rec_file = json.dumps(file_rec_dict, indent=4)
        with open(experiment_rec_dir, 'w') as outfile:
            outfile.write(experiment_rec_file)

    def experiment_tags(self) -> None:
        """Method that logs hyperparameters to Comet ML experiment tags.
        """
        experiment_tags_list = []
        # Log hyperparameters as experiment tags
        experiment_tags_list.extend(['use_mask=' + str(self.use_mask),
                                     'representation_name=' + str(self.representation_name),
                                     'representation_dir_name=' + str(self.representation_dir_name),
                                     'transform_name=' + str(self.transform_name),
                                     'upsample_mode=' + str(self.upsample_mode), 'scale_flag=' + str(self.scale_flag),
                                     'bn_flag=' + str(self.bn_flag), 'dropout_flag=' + str(self.dropout_flag),
                                     'use_ddp=' + str(self.use_ddp), 'rank=' + str(self.rank),
                                     'use_amp=' + str(self.use_amp), 'use_zero=' + str(self.use_zero),
                                     'batch_size=' + str(self.batch_size), 'nb_epochs=' + str(self.nb_epochs),
                                     'nb_steps_bin=' + str(self.nb_steps_bin),
                                     'loss_name=' + str(self.loss_name), 'loss_weight=' + str(self.loss_weight),
                                     'loss_bias=' + str(self.loss_bias),
                                     ])

        if 'SNN' in self.model_name:
            experiment_tags_list.extend(['truncated_bptt_ratio=' + str(self.truncated_bptt_ratio)])
            experiment_tags_list.extend(['use_intermediate_output=' + str(self.use_intermediate_output)])
            experiment_tags_list.extend(['surrogate_name=' + str(self.surrogate_name)])
            experiment_tags_list.extend(['surrogate_scale=' + str(self.surrogate_scale)])

        experiment_tags_list.extend(['run_id=' + str(os.environ.get('MASTER_ADDR'))])

        if self.hpc_flag:
            experiment_tags_list.extend(['job_id=' + str(os.environ.get('SLURM_JOB_ID'))])

        if self.scheduler_name is not None:
            experiment_tags_list.extend(['scheduler_name=' + str(self.scheduler_name)])
            if self.scheduler_name == 'OneCycleLR_':
                experiment_tags_list.extend(['max_lr=' + str(self.lr_scheduler_max_lr)])
        else:
            experiment_tags_list.extend(['lr=' + str(self.learning_rate)])

        if 'Adam' in self.optimizer_name:
            experiment_tags_list.extend(['betas=' + str(self.betas)])

        if self.clip_type is not None:
            experiment_tags_list.extend(['clip_type=' + str(self.clip_type)])
            experiment_tags_list.extend(['clip_value=' + str(self.clip_value)])

        if self.compute_representation:
            experiment_tags_list.extend(['compute_representation=' + str(self.compute_representation)])

        if self.pretrained_flag:
            experiment_tags_list.extend(['pretrained_flag=' + str(self.pretrained_flag)])

        if self.dropout_flag:
            experiment_tags_list.extend(['dropout_p=' + str(self.dropout_p)])

        if self.model_name == SNNstr.FCSNNstr and self.hidden_dim_list is not None:
            experiment_tags_list.extend(['layers_list=' + str(self.hidden_dim_list)])
        else:
            if self.hidden_channels_list is not None:
                experiment_tags_list.extend(['channels_list=' + str(self.hidden_channels_list)])
                experiment_tags_list.extend(['kernel_size=' + str(self.kernel_size)])
                experiment_tags_list.extend(['stride=' + str(self.stride)])
                experiment_tags_list.extend(['padding=' + str(self.padding)])
                experiment_tags_list.extend(['dilation=' + str(self.dilation)])
            experiment_tags_list.extend(['bias=' + str(self.bias)])
            experiment_tags_list.extend(['padding_mode=' + str(self.padding_mode)])
            experiment_tags_list.extend(['pooling_flag=' + str(self.pooling_flag)])
            if self.pooling_flag:
                experiment_tags_list.extend(['pooling_type=' + str(self.pooling_type)])
            experiment_tags_list.extend(['use_same_layer=' + str(self.use_same_layer)])

        if self.model_name in [SNNstr.UNetSNNstr, SNNstr.ResBottleneckUNetSNNstr, ANNstr.UNetstr,
                               ANNstr.ResBottleneckUNetstr]:
            experiment_tags_list.extend(['skip_connection_type=' + str(self.skip_connection_type)])
            if self.model_name in [SNNstr.ResBottleneckUNetSNNstr, ANNstr.ResBottleneckUNetstr]:
                experiment_tags_list.extend(['residual_skip_connection_type=' + str(self.residual_skip_connection_type)])
                experiment_tags_list.extend(['nb_residual_block=' + str(self.nb_residual_block)])

        if self.experiment:
            self.experiment.add_tags(experiment_tags_list)

    def _log_model_graph(self) -> None:
        """Method that logs model architecture to Comet ML experiment tags.
        """
        if self.use_ddp:
            if hasattr(self.model.module, 'init_state'):
                self.model.module.init_state(1)
            self.model.module.eval()
        else:
            if hasattr(self.model, 'init_state'):
                self.model.init_state(1)
            self.model.eval()

        with torch.no_grad():
            # Log model graph
            if self.experiment:
                self.experiment.set_model_graph(summary(self.model.module if self.use_ddp else self.model,
                                                        input_size=
                                                        (1, self.input_dim, self.nb_steps_bin)
                                                        if self.model_name == SNNstr.FCSNNstr
                                                        else (1, 1, self.input_dim, self.nb_steps_bin),
                                                        depth=4,
                                                        col_names=['kernel_size', 'output_size', 'num_params', 'mult_adds'],
                                                        row_settings=['var_names'], verbose=0, device=self.device,
                                                        cache_forward_pass=True,
                                                        ),
                                                overwrite=True
                                                )

            if self.use_ddp:
                self.model.module.init_rec()
            else:
                self.model.init_rec()

        if self.empty_cache:
            self._empty_cache()

    def prepare_data_manager(self) -> None:
        """Method that defines the dataloader instance.
        """

        print('From Rank: {}, ==> Preparing data loader managers...'.format(self.rank))

        # Create sampler
        if self.use_ddp:
            sampler_train = DistributedSampler(self.dataset_manager_train, num_replicas=dist.get_world_size(),
                                               rank=self.current_device, shuffle=True, seed=self.seed,
                                               drop_last=self.drop_last)
            sampler_valid = DistributedSampler(self.dataset_manager_valid, num_replicas=dist.get_world_size(),
                                               rank=self.current_device, shuffle=False, seed=self.seed,
                                               drop_last=False)
            sampler_test = DistributedSampler(self.dataset_manager_test, num_replicas=dist.get_world_size(),
                                              rank=self.current_device, shuffle=False, seed=self.seed,
                                              drop_last=False)

            # Set dataloader managers (shuffle must be False)
            self.dataloader_manager_train = DataLoader(self.dataset_manager_train, batch_size=self.batch_size_per_rank,
                                                       shuffle=(sampler_train is None), num_workers=self.num_workers,
                                                       pin_memory=self.pin_memory, drop_last=self.drop_last,
                                                       sampler=sampler_train, prefetch_factor=self.prefetch_factor,
                                                       worker_init_fn=self.seed_worker, generator=self.generator,)
            self.dataloader_manager_valid = DataLoader(self.dataset_manager_valid,
                                                       batch_size=self.batch_size_per_rank_eval,
                                                       shuffle=(sampler_valid is None), num_workers=self.num_workers,
                                                       pin_memory=self.pin_memory, drop_last=False,
                                                       sampler=sampler_valid, prefetch_factor=self.prefetch_factor,
                                                       worker_init_fn=self.seed_worker, generator=self.generator,)
            self.dataloader_manager_test = DataLoader(self.dataset_manager_test,
                                                      batch_size=self.batch_size_per_rank_eval,
                                                      shuffle=(sampler_test is None), num_workers=self.num_workers,
                                                      pin_memory=self.pin_memory, drop_last=False,
                                                      sampler=sampler_test, prefetch_factor=self.prefetch_factor,
                                                      worker_init_fn=self.seed_worker, generator=self.generator,)
        else:
            self.dataloader_manager_train = DataLoader(self.dataset_manager_train, batch_size=self.batch_size,
                                                       shuffle=True, num_workers=self.num_workers,
                                                       pin_memory=self.pin_memory, drop_last=self.drop_last,
                                                       sampler=None, prefetch_factor=self.prefetch_factor,
                                                       worker_init_fn=self.seed_worker, generator=self.generator,)
            self.dataloader_manager_valid = DataLoader(self.dataset_manager_valid, batch_size=self.batch_size,
                                                       shuffle=False, num_workers=self.num_workers,
                                                       pin_memory=self.pin_memory, drop_last=False,
                                                       sampler=None, prefetch_factor=self.prefetch_factor,
                                                       worker_init_fn=self.seed_worker, generator=self.generator,)
            self.dataloader_manager_test = DataLoader(self.dataset_manager_test, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=self.num_workers,
                                                      pin_memory=self.pin_memory, drop_last=False,
                                                      sampler=None, prefetch_factor=self.prefetch_factor,
                                                      worker_init_fn=self.seed_worker, generator=self.generator,)

    def prepare_model(self) -> None:
        """Method that defines the model instance.
        """

        print('From Rank: {}, ==> Preparing model...'.format(self.rank))

        # Define SNN model
        if self.model_name == SNNstr.FCSNNstr:
            self.model = FCSNN(input_dim=self.input_dim, hidden_dim_list=self.hidden_dim_list,
                               output_dim=self.output_dim, nb_steps=self.nb_steps_bin,
                               truncated_bptt_ratio=self.truncated_bptt_ratio, spike_fn=self.spike_fn,
                               neuron_model=self.neuron_model, neuron_parameters=self.neuron_parameters,
                               weight_init=self.weight_init, scale_flag=self.scale_flag,
                               scale_factor=self.scale_factor, bn_flag=self.bn_flag, dropout_flag=self.dropout_flag,
                               dropout_p=self.dropout_p, device=self.device, dtype=self.dtype)

        elif self.model_name == SNNstr.CSNNstr:
            self.model = CSNN(input_dim=self.input_dim, hidden_channels_list=self.hidden_channels_list,
                              output_dim=self.output_dim, kernel_size=self.kernel_size, stride=self.stride,
                              padding=self.padding, dilation=self.dilation, bias=self.bias, padding_mode=self.padding_mode,
                              pooling_flag=self.pooling_flag, pooling_type=self.pooling_type,
                              use_same_layer=self.use_same_layer, nb_steps=self.nb_steps_bin,
                              truncated_bptt_ratio=self.truncated_bptt_ratio, spike_fn=self.spike_fn,
                              neuron_model=self.neuron_model, neuron_parameters=self.neuron_parameters,
                              weight_init=self.weight_init, scale_flag=self.scale_flag, scale_factor=self.scale_factor,
                              bn_flag=self.bn_flag, dropout_flag=self.dropout_flag, dropout_p=self.dropout_p,
                              device=self.device, dtype=self.dtype)

        elif self.model_name == SNNstr.UNetSNNstr:
            self.model = UNetSNN(input_dim=self.input_dim, hidden_channels_list=self.hidden_channels_list,
                                 output_dim=self.output_dim, kernel_size=self.kernel_size, stride=self.stride,
                                 padding=self.padding, dilation=self.dilation, bias=self.bias, padding_mode=self.padding_mode,
                                 pooling_flag=self.pooling_flag, pooling_type=self.pooling_type,
                                 use_same_layer=self.use_same_layer, nb_steps=self.nb_steps_bin,
                                 truncated_bptt_ratio=self.truncated_bptt_ratio, spike_fn=self.spike_fn,
                                 neuron_model=self.neuron_model, neuron_parameters=self.neuron_parameters,
                                 weight_init=self.weight_init, upsample_mode=self.upsample_mode,
                                 scale_flag=self.scale_flag, scale_factor=self.scale_factor,
                                 bn_flag=self.bn_flag, dropout_flag=self.dropout_flag, dropout_p=self.dropout_p,
                                 device=self.device, dtype=self.dtype, skip_connection_type=self.skip_connection_type,
                                 use_intermediate_output=self.use_intermediate_output)
        elif self.model_name == SNNstr.ResBottleneckUNetSNNstr:
            self.model = ResBottleneckUNetSNN(input_dim=self.input_dim, hidden_channels_list=self.hidden_channels_list,
                                              output_dim=self.output_dim, kernel_size=self.kernel_size,
                                              stride=self.stride, padding=self.padding, dilation=self.dilation,
                                              bias=self.bias, padding_mode=self.padding_mode,
                                              pooling_flag=self.pooling_flag, pooling_type=self.pooling_type,
                                              use_same_layer=self.use_same_layer,
                                              nb_steps=self.nb_steps_bin, truncated_bptt_ratio=self.truncated_bptt_ratio,
                                              spike_fn=self.spike_fn, neuron_model=self.neuron_model,
                                              neuron_parameters=self.neuron_parameters, weight_init=self.weight_init,
                                              upsample_mode=self.upsample_mode,
                                              scale_flag=self.scale_flag, scale_factor=self.scale_factor,
                                              bn_flag=self.bn_flag, dropout_flag=self.dropout_flag,
                                              dropout_p=self.dropout_p, device=self.device, dtype=self.dtype,
                                              skip_connection_type=self.skip_connection_type,
                                              nb_residual_block=self.nb_residual_block,
                                              residual_skip_connection_type=self.residual_skip_connection_type,
                                              use_intermediate_output=self.use_intermediate_output,
                                              )
        elif self.model_name == ANNstr.CNNstr:
            self.model = CNN(input_dim=self.input_dim, hidden_channels_list=self.hidden_channels_list,
                              output_dim=self.output_dim, kernel_size=self.kernel_size, stride=self.stride,
                              padding=self.padding, dilation=self.dilation, bias=self.bias,
                              use_same_layer=self.use_same_layer, nb_steps=self.nb_steps_bin,
                              activation_fn=self.activation_fn, weight_init=self.weight_init,
                              upsample_mode=self.upsample_mode,
                              scale_flag=self.scale_flag, scale_factor=self.scale_factor, bn_flag=self.bn_flag,
                              dropout_flag=self.dropout_flag, dropout_p=self.dropout_p, device=self.device,
                              dtype=self.dtype)
        elif self.model_name == ANNstr.UNetstr:
            self.model = UNet(input_dim=self.input_dim, hidden_channels_list=self.hidden_channels_list,
                              output_dim=self.output_dim, kernel_size=self.kernel_size, stride=self.stride,
                              padding=self.padding, dilation=self.dilation, bias=self.bias,
                              use_same_layer=self.use_same_layer, nb_steps=self.nb_steps_bin,
                              activation_fn=self.activation_fn, weight_init=self.weight_init,
                              upsample_mode=self.upsample_mode,
                              scale_flag=self.scale_flag, scale_factor=self.scale_factor, bn_flag=self.bn_flag,
                              dropout_flag=self.dropout_flag, dropout_p=self.dropout_p, device=self.device,
                              dtype=self.dtype, skip_connection_type=self.skip_connection_type)
        elif self.model_name == ANNstr.ResBottleneckUNetstr:
            self.model = ResBottleneckUNet(input_dim=self.input_dim, hidden_channels_list=self.hidden_channels_list,
                                           output_dim=self.output_dim, kernel_size=self.kernel_size, stride=self.stride,
                                           padding=self.padding, dilation=self.dilation, bias=self.bias,
                                           use_same_layer=self.use_same_layer, nb_steps=self.nb_steps_bin,
                                           activation_fn=self.activation_fn, weight_init=self.weight_init,
                                           upsample_mode=self.upsample_mode,
                                           scale_flag=self.scale_flag, scale_factor=self.scale_factor,
                                           bn_flag=self.bn_flag, dropout_flag=self.dropout_flag,
                                           dropout_p=self.dropout_p, device=self.device, dtype=self.dtype,
                                           skip_connection_type=self.skip_connection_type,
                                           nb_residual_block=self.nb_residual_block,
                                           residual_skip_connection_type=self.residual_skip_connection_type)

        if torch.cuda.is_available():
            self.model.cuda()

        if self.empty_cache:
            self._empty_cache()

        if self.dtype == torch.half:
            self.model = self.model.to(dtype=self.dtype)

        if self.use_ddp:
            if self.bn_flag:
                # Convert BatchNorm layer to SyncBatchNorm to synchronize data statistics
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

            self.model = DDP(self.model, device_ids=[self.current_device])

        self._log_model_graph()

    def visualize_data(self, dist_flag: bool = True, show_flag: bool = False, verbose: bool = False, n: int = 50,
                       epsilon: float = 1e-8) -> None:
        """Method that plots data.

        Parameters
        ----------
        dist_flag: bool
            Boolean that indicates weather to compute input data distribution plot.
        show_flag: bool
            Boolean that indicates weather to display plots.
        verbose: bool
            Boolean that indicates weather to print specific output.
        n: int
            Number of data subset for data distribution computation.
        epsilon: float
            A small value to avoid computation error.
        """

        if self.representation_name == RepresentationName.lca:
            # Plot GT kernel
            self.plot_weights(show_flag=show_flag)

        # plot data
        self.plot_input_data(dataset_manager=self.dataset_manager_test, show_flag=show_flag, verbose=verbose)

        # Plot data distribution
        if dist_flag:
            epsilon = torch.tensor([epsilon])
            data_load_dict = {'valid': self.dataset_manager_valid}

            for data_load_name, dataset_manager in data_load_dict.items():
                data_dict = {'noisy': dataset_manager.tensor_noisyspeech_coefficients_dir,
                             'clean': dataset_manager.tensor_cleanspeech_coefficients_dir}

                random_index_array = random_index(n=n, high=dataset_manager.__len__())

                for data_type, coefficients_dir in data_dict.items():
                    speech_coefficients_list = []

                    for i in random_index_array:
                        speech_coefficients_list.append(
                            dataset_manager.get_coefficients(coefficients_dir, i).view(-1, dataset_manager.nb_steps))

                    fig_name = f'{data_type} {data_load_name} data magnitude'
                    speech_coefficients = torch.stack(speech_coefficients_list).view(-1).cpu()

                    if self.representation_name == RepresentationName.stft:
                        speech_coefficients = speech_coefficients[torch.nonzero(speech_coefficients, as_tuple=True)]
                    elif self.representation_name == RepresentationName.stft:
                        speech_coefficients = torch.abs(speech_coefficients)

                    self.visualization_manager.plot_dist(coefficients=speech_coefficients,
                                                         signal_type=fig_name, show_flag=show_flag,
                                                         fig_dir=os.path.join(self.plots_dir, fig_name))

                    if self.representation_name == RepresentationName.stft:
                        fig_name_t = fig_name.replace('magnitude', 'log-magnitude')
                        speech_coefficients_t = torch.log(torch.max(torch.pow(speech_coefficients, 2), epsilon))
                        self.visualization_manager.plot_dist(coefficients=speech_coefficients_t,
                                                             signal_type=fig_name_t, show_flag=show_flag,
                                                             fig_dir=os.path.join(self.plots_dir, fig_name_t))

    def _set_seed(self, seed: int = 1, deterministic: bool = False) -> None:
        """Method that sets seed.

        Parameters
        ----------
        seed: int
            Generator seed.
        deterministic: bool
            Boolean that indicates weather to use reproducible set.
        """

        torch.backends.cudnn.enabled = True

        if deterministic:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.random.manual_seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            # CUDA convolution determinism
            torch.backends.cudnn.deterministic = True

            # CUDA convolution benchmarking: False = cuDNN will deterministically select an algorithm
            torch.backends.cudnn.benchmark = False

            # Set a fixed value for the hash seed
            os.environ["PYTHONHASHSEED"] = str(seed)

            # Generator
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            self.generator = None

    @staticmethod
    def _set_seed_worker(seed: int = 1) -> None:
        """Method that sets DataLoader worker seed.

        Parameters
        ----------
        seed: int
            Generator seed.
        """
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    @staticmethod
    def _set_debug_apis(debug_api: bool = False) -> None:
        """Method that enables/disables debug APIs.

        Parameters
        ----------
        debug_api: bool
            Boolean that indicates weather to enable/disable debug APIs.
        """
        torch.autograd.set_detect_anomaly(debug_api)        # anomaly detection
        torch.autograd.profiler.profile(debug_api)          # autogradprofiler
        torch.autograd.profiler.emit_nvtx(debug_api)        # automatic NVTX ranges

    @staticmethod
    def _empty_cache() -> None:
        """Method that empties cache and runs the garbage collector.
        """
        torch.cuda.empty_cache()
        # Garbage collection
        gc.collect()

    def __call__(self) -> None:

        print('From Rank: {}, ==> Plot input data samples...'.format(self.rank))
        self.visualize_data(dist_flag=False, show_flag=False)

        if self.empty_cache:
            self._empty_cache()

        # Initialize results dict
        results_dc = {}

        # -----------------------------------------        Train model         -----------------------------------------
        if self.train_flag:
            print('From Rank: {}, ==> Start training...'.format(self.rank))

            if self.empty_cache:
                self._empty_cache()

            # Comet_ml experiment tags
            self.experiment_tags()

            # Prepare Data loader manager
            self.prepare_data_manager()

            # Define model
            self.prepare_model()

            self.model_train_manager = TrainValidTestManager(model=self.model,
                                                             model_name=self.model_name,
                                                             model_file_dir=self.model_file_dir,
                                                             exp_model_file_dir=self.exp_model_file_dir,
                                                             use_mask=self.use_mask,
                                                             dataloader_manager=self.dataloader_manager_train,
                                                             dataloader_manager_valid=self.dataloader_manager_valid,
                                                             representation_name=self.representation_name,
                                                             transform_manager=self.dataset_manager_train.transform_manager,
                                                             visualization_manager=self.visualization_manager,
                                                             weights=self.dataset_manager_train.get_weights()
                                                             if self.representation_name == RepresentationName.lca
                                                             else None,
                                                             loss_name=self.loss_name,
                                                             loss_weight=self.loss_weight,
                                                             loss_bias=self.loss_bias,
                                                             optimizer_name=self.optimizer_name,
                                                             learning_rate=self.learning_rate,
                                                             betas=self.betas,
                                                             scheduler_name=self.scheduler_name,
                                                             lr_scheduler_max_lr=self.lr_scheduler_max_lr,
                                                             lr_scheduler_gamma=self.lr_scheduler_gamma,
                                                             clip_type=self.clip_type,
                                                             clip_value=self.clip_value,
                                                             nb_epochs=self.nb_epochs,
                                                             nb_warmup_epochs=self.nb_warmup_epochs,
                                                             nb_steps=self.dataset_manager_train.nb_steps,
                                                             nb_steps_bin=self.nb_steps_bin,
                                                             experiment=self.experiment,
                                                             train_neuron_parameters=
                                                             self.neuron_parameters['train_neuron_parameters'],
                                                             save_mem=self.save_mem,
                                                             save_model=self.save_model,
                                                             log_hist_flag=True,
                                                             use_ddp=self.use_ddp,
                                                             device=self.device,
                                                             dtype=self.dtype,
                                                             evaluate_flag=self.evaluate_flag,
                                                             use_amp=self.use_amp,
                                                             use_zero=self.use_zero,
                                                             empty_cache=self.empty_cache,
                                                             pretrained_flag=self.pretrained_flag,
                                                             perceptual_metric_flag=self.perceptual_metric_flag
                                                             )

            start = datetime.now()

            self.model_train_manager.train_model()

            training_duration = str(datetime.now() - start)
            print(f'- Training duration = {training_duration}')
            if self.experiment:
                self.experiment.log_other(key='Training duration', value=training_duration)

            results_dc['Training duration'] = training_duration
            results_dc['Training loss'] = self.model_train_manager.results['Training loss']
            results_dc['Validation loss'] = self.model_train_manager.results['Validation loss']

            print('From Rank: {}, ==> Training complete...'.format(self.rank))

            if self.rank == 0:
                evaluation_show_flag = False

                # Set plot figure saving path
                loss_plot_fig_dir = os.path.join(self.experiment_files_dir, 'train_valid_loss')

                # Plot training loss
                self.visualization_manager.plot_loss(
                    training_loss_hist=self.model_train_manager.results['Training loss'],
                    validation_loss_hist=self.model_train_manager.results['Validation loss'],
                    show_flag=evaluation_show_flag,
                    fig_dir=loss_plot_fig_dir)

                if self.perceptual_metric_flag:
                    perceptual_metric_plot_fig_dir = os.path.join(self.experiment_files_dir,
                                                                  'train_valid_perceptual_metric')
                    self.visualization_manager.plot_perceptual_metric(
                        training_perceptual_metric_hist=self.model_train_manager.results[
                            f'Training {self.model_train_manager.perceptual_metric_name}'],
                        validation_perceptual_metric_hist=self.model_train_manager.results[
                            f'Validation {self.model_train_manager.perceptual_metric_name}'],
                        show_flag=evaluation_show_flag,
                        fig_dir=perceptual_metric_plot_fig_dir)

                # Save the trained model used for the current experiment
                self.model_train_manager.save_checkpoint(epoch=self.nb_epochs,
                                                         filepath=self.model_file_dir,
                                                         log_model_flag=True)

            if self.use_ddp:
                dist.barrier()

        # -----------------------------------------         Test model         -----------------------------------------

        print('From Rank: {}, ==> Start testing...'.format(self.rank))

        if self.empty_cache:
            self._empty_cache()

        if not self.train_flag:
            # Comet_ml experiment tags
            self.experiment_tags()

            # Prepare Data loader manager
            self.prepare_data_manager()

            # Define model
            self.prepare_model()

            self.pretrained_flag = True
            self.save_model = True
        else:
            self.pretrained_flag = False
            self.save_model = False

        if self.empty_cache:
            self._empty_cache()

        self.model_test_manager = TrainValidTestManager(model=self.model,
                                                        model_name=self.model_name,
                                                        model_file_dir=self.model_file_dir,
                                                        exp_model_file_dir=self.exp_model_file_dir,
                                                        use_mask=self.use_mask,
                                                        dataloader_manager=self.dataloader_manager_test,
                                                        dataloader_manager_valid=None,
                                                        representation_name=self.representation_name,
                                                        transform_manager=self.dataset_manager_test.transform_manager,
                                                        visualization_manager=self.visualization_manager,
                                                        weights=self.dataset_manager_test.get_weights()
                                                        if self.representation_name == RepresentationName.lca else None,
                                                        loss_name=self.loss_name,
                                                        loss_weight=self.loss_weight,
                                                        loss_bias=self.loss_bias,
                                                        optimizer_name=self.optimizer_name,
                                                        learning_rate=self.learning_rate,
                                                        betas=self.betas,
                                                        scheduler_name=self.scheduler_name,
                                                        lr_scheduler_max_lr=self.lr_scheduler_max_lr,
                                                        lr_scheduler_gamma=self.lr_scheduler_gamma,
                                                        clip_type=self.clip_type,
                                                        clip_value=self.clip_value,
                                                        nb_epochs=self.nb_epochs,
                                                        nb_warmup_epochs=self.nb_warmup_epochs,
                                                        nb_steps=self.dataset_manager_test.nb_steps,
                                                        nb_steps_bin=self.nb_steps_bin,
                                                        experiment=self.experiment,
                                                        train_neuron_parameters=
                                                        self.neuron_parameters['train_neuron_parameters'],
                                                        save_mem=True,
                                                        save_model=self.save_model,
                                                        log_hist_flag=True,
                                                        use_ddp=self.use_ddp,
                                                        device=self.device,
                                                        dtype=self.dtype,
                                                        use_amp=self.use_amp,
                                                        use_zero=self.use_zero,
                                                        empty_cache=self.empty_cache,
                                                        pretrained_flag=self.pretrained_flag,
                                                        perceptual_metric_flag=self.perceptual_metric_flag
                                                        )

        # Test model
        start = datetime.now()

        self.model_test_manager.test_model(
            tensor_enhanced_coefficients_dir=self.dataset_manager_test.tensor_enhanced_coefficients_dir,
            coefficients_filename=self.dataset_manager_test.coefficients_filename)

        # Write testing duration to json file
        test_duration = str(datetime.now() - start)
        print(f'- Test duration = {test_duration}')
        if self.experiment:
            self.experiment.log_other(key='Test duration', value=test_duration)

        print('From Rank: {}, ==> Testing complete...'.format(self.rank))

        results_dc['Test duration'] = test_duration
        results_dc['Test loss'] = self.model_test_manager.results['Test loss']

        # Save the trained model used for the current experiment
        if not self.train_flag:
            if self.rank == 0:
                self.model_test_manager.save_checkpoint(epoch=self.nb_epochs,
                                                        filepath=self.exp_model_file_dir,
                                                        log_model_flag=True)
            if self.use_ddp:
                dist.barrier()

        # Reconstruct audio signal from enhanced activation
        print('Reconstruct audio...')
        if self.rank == 0:
            self.dataset_manager_test.audio_reconstruction()
        if self.use_ddp:
            dist.barrier()

        print('From Rank: {}, ==> Plot output data samples...'.format(self.rank))
        self.plot_output_data(dataset_manager=self.dataset_manager_test)

        print('Compute audio evaluation metrics...')

        if self.rank == 0:
            evaluation_manager = EvaluationManager(dataset_manager=self.dataset_manager_test,
                                                   experiment_files_dir=self.experiment_files_dir,
                                                   experiment=self.experiment,
                                                   use_representation_dir=False,
                                                   )
            evaluation_manager.compute_evaluation_metrics()

            # Save results to json file
            self.save_json_file(file_name='results',
                                file_rec_dict={
                                    'Train/Test results': results_dc,
                                    'OBJECTIVE evaluation': evaluation_manager.obj_evaluation_dc,
                                    'COMPOSITE OBJECTIVE evaluation': evaluation_manager.composite_obj_evaluation_dc,
                                    'DNSMOS OBJECTIVE evaluation': evaluation_manager.dnsmos_obj_evaluation_dc})

        if self.use_ddp:
            dist.barrier()

        print('Done...')

        # if self.use_ddp:
        #     self._cleanup()

        # Indicate that the experiment is complete
        # if self.experiment:
        #     self.experiment.end()


