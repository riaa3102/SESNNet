"""
File:
    constants.py

Description:
    This file stores helpful constants value.
"""

from enum import Enum


class Default(Enum):

    representation_name = 'stft'                        # 'stft', 'lca'
    representation_dir_name = 'STFT_4s_nfft=512_wl=512_hl=256'
    transform_name = ['log_power', 'standardize']       # 'maxabs', 'quantile_maxabs', 'normalize', 'standardize',
                                                        # 'shift', 'log', 'log_power', 'log10'

    # ----------------------------------------------------------------------------------

    # --------------------
    # 0.03      =>  0.97
    # 0.01      =>  0.91
    # 0.00145   =>  0.5
    # 0.00044   =>  0.1
    # 0.00034   =>  0.05
    # 0.0002    =>  0.01
    # 0.0001    =>  0.
    # --------------------

    time_step = 0.001
    membrane_threshold = 1.
    k = 1.
    tau = 20.
    tau_syn = 0.00034
    tau_mem = 0.00034
    tau_syn_out = 0.00034
    tau_mem_out = 0.00034

    spiking_mode = 'binary'                             # None, 'binary', 'graded'
    reset_mode = 'soft_reset'                           # 'hard_reset', 'soft_reset'

    # ----------------------------------------------------------------------------------

    padding_mode = 'zeros'
    pooling_type = 'max'
    weight_mean = 0.
    weight_std = 0.2                                    # ANN: 0.02, SNN: 0.2
    weight_gain = 5.
    weight_init_dist = 'normal_'       # 'normal_', 'uniform_', 'kaiming_uniform_', 'kaiming_normal_', 'xavier_uniform_'
    weight_mul_factor = None
    neuron_parameters_init_dist = 'normal_'             # 'constant_', 'normal_', 'uniform_'
    neuron_model = 'lif'
    upsample_mode = 'nearest'                           # 'nearest', 'bilinear'
    dropout_p = 0.1
    scale_factor = 1.
    skip_connection_type = 'cat_'                       # 'cat_', 'add_'
    nb_residual_block = 1
    residual_skip_connection_type = 'add_'              # 'add_', 'and_', 'iand_'

    # ----------------------------------------------------------------------------------

    # SNN
    input_dim = 256

    # FCSNN
    hidden_dim_list = None

    # CSNN
    hidden_channels_list = None
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = None
    dilation = (1, 1)

    # ----------------------------------------------------------------------------------

    nb_epochs = 30
    nb_warmup_epochs = 1
    batch_size = 32
    nb_steps_bin = None
    truncated_bptt_ratio = None
    surrogate_name = 'ATan'                 # 'SuperSpike', 'SigmoidDerivative', 'ATan', 'PiecewiseLinear', 'SpikeFunc'
    surrogate_scale = None
    activation_fn = ['lrelu']                       # 'sigmoid', 'relu', 'lrelu', 'prelu', 'tanh'
    loss_name = ['mse_loss']                        # 'mse_loss', 'time_mse_loss', 'differentiable_stoi_loss',
                                                    # 'differentiable_si_sdr_loss'
    loss_weight = [1.]
    loss_bias = [0.]
    optimizer_name = 'Adam'                         # 'Adam', 'AdamW', 'RAdam', 'RMSProp'
    learning_rate = 2e-4
    betas = (0.5, 0.9)                              # (0.9, 0.999), (0.5, 0.9)
    clip_type = None
    clip_value = 10.
    scheduler_name = None                           # None, 'OneCycleLR', 'StepLR', 'MultiStepLR', 'ExponentialLR'
    lr_scheduler_max_lr = 0.002
    lr_scheduler_gamma = 0.8
    model_file_name_id = None

    # ----------------------------------------------------------------------------------

    workspace = None
    api_key = None
    project_name = None

    # ----------------------------------------------------------------------------------

    dist_backend = 'nccl'                           # 'nccl', 'gloo', 'mpi'
    num_workers = 0
    prefetch_factor = 2
    seed = 1


