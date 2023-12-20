"""
File:
    main.py

Description:
    Training and testing the model: main script.
"""

import os
from pathlib import Path
import argparse
import yaml
from src.model.SpeechEnhancer import SpeechEnhancer
from src.model.constants import SNNstr, ANNstr
from constants import Default


def argument_parser():
    """
    This function defines a parser to enable user to run different configurations using terminal
    """

    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python main.py',
                                     description='This program enables user to train/test different models of speech '
                                                 'enhancement using spiking neural networks.')

    # --------------------------------------       Setting model parameters       --------------------------------------
    parser.add_argument('-m', '--model_name', type=str, default=SNNstr.UNetSNNstr,
                        choices=[SNNstr.FCSNNstr, SNNstr.CSNNstr, SNNstr.UNetSNNstr, SNNstr.ResBottleneckUNetSNNstr,
                                 ANNstr.CNNstr, ANNstr.UNetstr, ANNstr.ResBottleneckUNetstr],
                        help=f'Name of the model')
    parser.add_argument('-msk', '--use_mask', action='store_true',
                        help='Boolean that indicates weather to use direct approach or masking approach')
    parser.add_argument('-rec', '--reconstruct_flag', action='store_true',
                        help='Boolean that indicates weather to enhance noisy speech or reconstruct clean speech')

    # --------------------------------------       Setting data parameters        --------------------------------------
    parser.add_argument('-rn', '--representation_name', type=str, default=Default.representation_name.value,
                        help='Representation name')
    parser.add_argument('-rdn', '--representation_dir_name', type=str, default=Default.representation_dir_name.value,
                        help='Representation directory name')
    parser.add_argument('-tn', '--transform_name', nargs='*', type=str, default=Default.transform_name.value,
                        choices=['maxabs', 'quantile_maxabs', 'normalize', 'standardize', 'shift', 'log_power', 'log10'],
                        help='List that contains data transform name')
    parser.add_argument('-cr', '--compute_representation', action='store_true',
                        help='Boolean that indicates weather to compute data')

    # --------------------------------------    Setting SNN neuron parameters     --------------------------------------
    parser.add_argument('--k', type=float, default=Default.k.value, help='KLIF neuron model k constant')
    parser.add_argument('--tau', type=float, default=Default.tau.value, help='PLIF neuron model time constant')
    parser.add_argument('--tau_syn', type=float, default=Default.tau_syn.value,
                        help='Spiking layer input current time constant')
    parser.add_argument('--tau_mem', type=float, default=Default.tau_mem.value,
                        help='Spiking layer membrane potential time constant')
    parser.add_argument('--tau_syn_out', type=float, default=Default.tau_syn_out.value,
                        help='Readout layer input current time constant')
    parser.add_argument('--tau_mem_out', type=float, default=Default.tau_mem_out.value,
                        help='Readout layer membrane potential time constant')
    parser.add_argument('--time_step', type=float, default=Default.time_step.value,
                        help='SNN time step')
    parser.add_argument('--membrane_threshold', type=float, default=Default.membrane_threshold.value,
                        help='Spiking layer membrane threshold')
    parser.add_argument('-di', '--decay_input', action='store_true',
                        help='Boolean that indicates weather the input will decay')
    parser.add_argument('-spk', '--spiking_mode', type=str, default=Default.spiking_mode.value,
                        choices=['binary', 'graded'],
                        help='Spiking mode')
    parser.add_argument('-rst', '--reset_mode', type=str, default=Default.reset_mode.value,
                        choices=['hard_reset', 'soft_reset'],
                        help='Reset mode: reset to zero (hard_reset) or reset by subtraction (soft_reset)')
    parser.add_argument('-dr', '--detach_reset', action='store_true',
                        help='Boolean that indicates weather to detach the computation graph of reset term in backward')

    # ---------------------------------  Setting SNN weight initialization parameters  ---------------------------------
    parser.add_argument('-wm', '--weight_mean', type=float, default=Default.weight_mean.value,
                        help='Mean of weight initialization')
    parser.add_argument('-ws', '--weight_std', type=float, default=Default.weight_std.value,
                        help='Standard deviation of weight initialization')
    parser.add_argument('-wg', '--weight_gain', type=float, default=Default.weight_gain.value,
                        help='Gain of weight initialization')
    parser.add_argument('-wi', '--weight_init_dist', type=str, default=Default.weight_init_dist.value,
                        choices=['normal_', 'uniform_', 'kaiming_normal_', 'kaiming_uniform_', 'xavier_uniform_'],
                        help='Weight initialization distribution')

    # --------------------------------------     Setting model architecture      ---------------------------------------
    parser.add_argument('-inp', '--input_dim', type=int, default=Default.input_dim.value,
                        help='Input layer dimension')
    parser.add_argument('-hl', '--hidden_dim_list', nargs='*', type=int, default=Default.hidden_dim_list.value,
                        help='List of hidden layers (linear layer) dimension')
    parser.add_argument('-hc', '--hidden_channels_list', nargs='*', type=int,
                        default=Default.hidden_channels_list.value,
                        help='List of hidden layers (convolutional layer) channels dimension')
    parser.add_argument('-k', '--kernel_size', nargs='*', type=int, default=Default.kernel_size.value,
                        help='List of convolutional layers kernel size')
    parser.add_argument('-s', '--stride', nargs='*', type=int, default=Default.stride.value,
                        help='List of convolutional layers stride')
    parser.add_argument('-p', '--padding', nargs='*', type=int, default=Default.padding.value, required=False,
                        help='List of convolutional layers padding')
    parser.add_argument('-d', '--dilation', nargs='*', type=int, default=Default.dilation.value,
                        help='List of convolutional layers dilation')
    parser.add_argument('-bs', '--bias', action='store_true',
                        help='Boolean that indicates weather to use a bias term for convolutional layers')
    parser.add_argument('-cpm', '--padding_mode', type=str, default=Default.padding_mode.value,
                        choices=['zeros', 'reflect', 'replicate', 'circular'],
                        help='Convolutional layers padding mode')
    parser.add_argument('-pf', '--pooling_flag', action='store_true',
                        help='Boolean that indicates weather to use a pooling layer for downsampling')
    parser.add_argument('-pt', '--pooling_type', type=str, default=Default.pooling_type.value,
                        choices=['max', 'avg'],
                        help='Pooling layer type: max, avg')
    parser.add_argument('-sl', '--use_same_layer', action='store_true',
                        help='Boolean that indicates weather to add layers (input and output layers) with same shape')

    parser.add_argument('-r', '--recurrent_flag', action='store_true',
                        help='Boolean that indicates weather to add recurrence term to the input current equation')
    parser.add_argument('-nm', '--neuron_model', type=str, default=Default.neuron_model.value,
                        choices=['lif', 'plif', 'if'],
                        help='Neuron model name')
    parser.add_argument('-tnp', '--train_neuron_parameters', action='store_true',
                        help='Boolean that indicates weather to train neuron parameters')
    parser.add_argument('-npd', '--neuron_parameters_init_dist', type=str,
                        default=Default.neuron_parameters_init_dist.value,
                        choices=['constant_', 'normal_', 'uniform_'],
                        help='Neuron model parameters initialization distribution')
    parser.add_argument('-up', '--upsample_mode', type=str, default=Default.upsample_mode.value,
                        choices=['nearest', 'bilinear'],
                        help='Upsampling mode: nearest, bilinear')
    parser.add_argument('-tmf', '--scale_flag', action='store_true',
                        help='Boolean that indicates weather to train the scaling layer')
    parser.add_argument('-mf', '--scale_factor', type=float, default=Default.scale_factor.value,
                        help='Constant value for the scaling layer')
    parser.add_argument('-bnf', '--bn_flag', action='store_true',
                        help='Boolean that indicates weather to add a batch normalization layer')
    parser.add_argument('-df', '--dropout_flag', action='store_true',
                        help='Boolean that indicates weather to add a dropout layer')
    parser.add_argument('-dp', '--dropout_p', type=float, default=Default.dropout_p.value,
                        help='Dropout probability of an element to be zeroed')
    parser.add_argument('-skp', '--skip_connection_type', type=str, default=Default.skip_connection_type.value,
                        choices=['cat_', 'add_'],
                        help='Skip connections type')
    parser.add_argument('-rsd', '--nb_residual_block', type=int, default=Default.nb_residual_block.value,
                        help='Number of transitional blocks for the residual block')
    parser.add_argument('-rskp', '--residual_skip_connection_type', type=str,
                        default=Default.residual_skip_connection_type.value,
                        choices=['add_', 'and_', 'iand_'],
                        help='Residual skip connections type')
    parser.add_argument('-out', '--use_intermediate_output', action='store_true',
                        help='Boolean that indicates weather to add rescaled output from intermediate layers')

    # --------------------------------------     Setting training parameters      --------------------------------------
    parser.add_argument('-e', '--nb_epochs', type=int, default=Default.nb_epochs.value,
                        help='Number of training iterations')
    parser.add_argument('-we', '--nb_warmup_epochs', type=int, default=Default.nb_warmup_epochs.value,
                        help=f'Number of training warmup iterations')
    parser.add_argument('-b', '--batch_size', type=int, default=Default.batch_size.value,
                        help=f'Number of data samples per batch')
    parser.add_argument('-ts', '--nb_steps_bin', type=int, default=Default.nb_steps_bin.value, required=False,
                        help='Number of forward pass time steps per bin')
    parser.add_argument('-tbptt', '--truncated_bptt_ratio', type=int, default=Default.truncated_bptt_ratio.value, required=False,
                        help='Truncated Backpropagation Through Time (BPTT) ratio over time steps')
    parser.add_argument('-surr', '--surrogate_name', type=str, default=Default.surrogate_name.value,
                        choices=['SuperSpike', 'SigmoidDerivative', 'ATan', 'PiecewiseLinear'],
                        help='Surrogate gradient function name')
    parser.add_argument('-ss', '--surrogate_scale', type=float, default=Default.surrogate_scale.value,
                        help='Surrogate gradient scale')
    parser.add_argument('-act', '--activation_fn', nargs='*', type=str, default=Default.activation_fn.value,
                        choices=[None, 'sigmoid', 'relu', 'lrelu', 'prelu', 'tanh'],
                        help='List of ANN activation function name: input, hidden, output layers')
    parser.add_argument('-ln', '--loss_name', nargs='*', type=str, default=Default.loss_name.value,
                        choices=['mse_loss', 'l1_loss', 'lsd_loss', 'time_mse_loss', 'huber_loss', 'stoi_loss',
                                 'si_snr_loss', 'si_sdr_loss'],
                        help='List of loss function name')
    parser.add_argument('-lw', '--loss_weight', nargs='*', type=float, default=Default.loss_weight.value,
                        help='List of loss function weight')
    parser.add_argument('-lb', '--loss_bias', nargs='*', type=float, default=Default.loss_bias.value,
                        help='List of loss function bias')
    parser.add_argument('-o', '--optimizer_name', type=str, default=Default.optimizer_name.value,
                        help='Optimizer name')
    parser.add_argument('-lr', '--learning_rate', type=float, default=Default.learning_rate.value,
                        help='Learning rate of the model during training')
    parser.add_argument('-bt', '--betas', nargs='*', type=float, default=Default.betas.value,
                        help='Optimizer betas parameter')

    parser.add_argument('-sch', '--scheduler_name', type=str, default=Default.scheduler_name.value,
                        choices=['OneCycleLR', 'StepLR', 'MultiStepLR', 'ExponentialLR'],
                        help='Scheduler name')
    parser.add_argument('-lrm', '--lr_scheduler_max_lr', type=float, default=Default.lr_scheduler_max_lr.value,
                        help='Learning rate scheduler max_lr parameter')
    parser.add_argument('-lrg', '--lr_scheduler_gamma', type=float, default=Default.lr_scheduler_gamma.value,
                        help='Learning rate scheduler gamma parameter')

    parser.add_argument('-ct', '--clip_type', type=str, default=Default.clip_type.value,
                        choices=[None, 'value', 'norm'],
                        help='Gradient clipping type')
    parser.add_argument('-cv', '--clip_value', type=float, default=Default.clip_value.value,
                        help='Gradient clipping value')

    parser.add_argument('-tr', '--train_flag', action='store_true',
                        help='Boolean that indicates weather to train the model')
    parser.add_argument('-pr', '--pretrained_flag', action='store_true',
                        help='Boolean that indicates weather to load a pretrained model')
    parser.add_argument('-mid', '--model_file_name_id', type=int, default=Default.model_file_name_id.value,
                        help='Pretrained model file id')
    parser.add_argument('-sme', '--save_mem', action='store_true',
                        help='Boolean that indicates weather to log hidden activations')
    parser.add_argument('-smd', '--save_model', action='store_true',
                        help='Boolean that indicates weather to save checkpoint of pretrained model')
    parser.add_argument('-pmf', '--perceptual_metric_flag', action='store_true',
                        help='Boolean that indicates weather to compute a perceptual metric')
    parser.add_argument('-dbg', '--debug_flag', action='store_true',
                        help='Boolean that indicates weather to use debugging dataset')

    # ---------------------------------------------        Comet ML        ---------------------------------------------
    parser.add_argument('--workspace', type=str, default=Default.workspace.value,
                        help='Comet ML experiment instance workspace argument')
    parser.add_argument('--api_key', type=str, default=Default.api_key.value,
                        help='Comet ML experiment instance api_key argument')
    parser.add_argument('--project_name', type=str, default=Default.project_name.value,
                        help='Comet ML experiment instance project_name argument')

    # ------------------------------------------        Running machine        -----------------------------------------
    parser.add_argument('-hpc', '--hpc_flag', action='store_true',
                        help='Boolean that indicates weather to use HPC configuration')
    parser.add_argument('-dist', '--use_ddp', action='store_true',
                        help='Boolean that indicates weather to use Pytorch Distributed Data Parallel (DDP) library')
    parser.add_argument('-db', '--dist_backend', type=str, default=Default.dist_backend.value,
                        choices=['nccl', 'gloo'],
                        help='DDP backend name')
    parser.add_argument('-eval', '--evaluate_flag', action='store_true',
                        help='Boolean that indicates weather to evaluate model using the validation set during training')
    parser.add_argument('-cast', '--use_amp', action='store_true',
                        help='Boolean that indicates weather to use Pytorch Automatic Mixed Precision (AMP) library')
    parser.add_argument('-zero', '--use_zero', action='store_true',
                        help='Boolean that indicates weather to use Pytorch Zero Redundancy Optimizer (ZeRO)')
    parser.add_argument('-pm', '--pin_memory', action='store_true',
                        help='Boolean that indicates pin_memory parameter for the data loader')
    parser.add_argument('-nw', '--num_workers', type=int, default=Default.num_workers.value,
                        help='Number of workers parameter for the data loader')
    parser.add_argument('-prf', '--prefetch_factor', type=int, default=Default.prefetch_factor.value,
                        help='Prefetch factor parameter for the data loader')

    parser.add_argument('-det', '--deterministic', action='store_true',
                        help='Boolean that indicates weather to use deterministic mode for reproducibility')
    parser.add_argument('-sd', '--seed', type=int, default=Default.seed.value,
                        help='Reproducibility seed')
    parser.add_argument('-ec', '--empty_cache', action='store_true',
                        help='Boolean that indicates weather to empty cache')

    return parser


def main():
    # Parse arguments
    args = argument_parser().parse_args()

    # ---------------------------------------------------   Debug   ----------------------------------------------------
    local_train_flag = False
    # ------------------------------------------------------------------------------------------------------------------
    if local_train_flag:
        args.compute_representation = False
        args.model_name = 'UNet'
        args.use_mask = False
        args.pretrained_flag = False
        args.train_flag = True
        args.nb_epochs = 3
        args.batch_size = 4
        args.learning_rate = 0.0004
        args.train_neuron_parameters = True
        args.recurrent_flag = True
        args.detach_reset = True
        args.use_ddp = True
        args.pin_memory = True
        args.num_workers = 0
        args.empty_cache = True
        args.evaluate_flag = True
        args.loss_name = ['lsd_loss']
        args.loss_weight = [1.]
        args.loss_bias = [0.]
        args.perceptual_metric_flag = True
        args.save_mem = True
        args.deterministic = True
        args.debug_flag = True
    # ------------------------------------------------------------------------------------------------------------------

    # Print arguments
    print('\nMain file arguments...')
    print('----------------------')

    if args.transform_name == 'None':
        args.transform_name = None

    for i in range(len(args.activation_fn)):
        if args.activation_fn[i] == 'None':
            args.activation_fn[i] = None

    for arg in vars(args):
        print('\t - {}: {}'.format(arg, getattr(args, arg)))

    comet_ml_params = {'api_key': args.api_key,
                       'workspace': args.workspace,
                       'project_name': args.project_name,
                       }

    # Speech Enhancement model
    speech_enhancer = SpeechEnhancer(model_name=args.model_name,
                                     use_mask=args.use_mask,
                                     representation_name=args.representation_name,
                                     representation_dir_name=args.representation_dir_name,
                                     transform_name=args.transform_name,
                                     reconstruct_flag=args.reconstruct_flag,
                                     compute_representation=args.compute_representation,
                                     batch_size=args.batch_size,
                                     k=args.k,
                                     tau=args.tau,
                                     tau_syn=args.tau_syn,
                                     tau_mem=args.tau_mem,
                                     tau_syn_out=args.tau_syn_out,
                                     tau_mem_out=args.tau_mem_out,
                                     time_step=args.time_step,
                                     membrane_threshold=args.membrane_threshold,
                                     decay_input=args.decay_input,
                                     spiking_mode=args.spiking_mode,
                                     reset_mode=args.reset_mode,
                                     detach_reset=args.detach_reset,
                                     weight_mean=args.weight_mean,
                                     weight_std=args.weight_std,
                                     weight_gain=args.weight_gain,
                                     weight_init_dist=args.weight_init_dist,
                                     input_dim=args.input_dim,
                                     hidden_dim_list=args.hidden_dim_list,
                                     hidden_channels_list=args.hidden_channels_list,
                                     kernel_size=args.kernel_size,
                                     stride=args.stride,
                                     padding=args.padding,
                                     dilation=args.dilation,
                                     bias=args.bias,
                                     padding_mode=args.padding_mode,
                                     pooling_flag=args.pooling_flag,
                                     pooling_type=args.pooling_type,
                                     use_same_layer=args.use_same_layer,
                                     recurrent_flag=args.recurrent_flag,
                                     neuron_model=args.neuron_model,
                                     train_neuron_parameters=args.train_neuron_parameters,
                                     neuron_parameters_init_dist=args.neuron_parameters_init_dist,
                                     upsample_mode=args.upsample_mode,
                                     scale_flag=args.scale_flag,
                                     scale_factor=args.scale_factor,
                                     bn_flag=args.bn_flag,
                                     dropout_flag=args.dropout_flag,
                                     dropout_p=args.dropout_p,
                                     skip_connection_type=args.skip_connection_type,
                                     nb_residual_block=args.nb_residual_block,
                                     residual_skip_connection_type=args.residual_skip_connection_type,
                                     use_intermediate_output=args.use_intermediate_output,
                                     loss_name=args.loss_name,
                                     loss_weight=args.loss_weight,
                                     loss_bias=args.loss_bias,
                                     surrogate_name=args.surrogate_name,
                                     surrogate_scale=args.surrogate_scale,
                                     activation_fn=args.activation_fn,
                                     optimizer_name=args.optimizer_name,
                                     learning_rate=args.learning_rate,
                                     betas=args.betas,
                                     scheduler_name=args.scheduler_name,
                                     lr_scheduler_max_lr=args.lr_scheduler_max_lr,
                                     lr_scheduler_gamma=args.lr_scheduler_gamma,
                                     clip_type=args.clip_type,
                                     clip_value=args.clip_value,
                                     nb_epochs=args.nb_epochs,
                                     nb_warmup_epochs=args.nb_warmup_epochs,
                                     nb_steps_bin=args.nb_steps_bin,
                                     truncated_bptt_ratio=args.truncated_bptt_ratio,
                                     pretrained_flag=args.pretrained_flag,
                                     model_file_name_id=args.model_file_name_id,
                                     save_mem=args.save_mem,
                                     save_model=args.save_model,
                                     perceptual_metric_flag=args.perceptual_metric_flag,
                                     train_flag=args.train_flag,
                                     hpc_flag=args.hpc_flag,
                                     use_ddp=args.use_ddp,
                                     dist_backend=args.dist_backend,
                                     evaluate_flag=args.evaluate_flag,
                                     use_amp=args.use_amp,
                                     use_zero=args.use_zero,
                                     pin_memory=args.pin_memory,
                                     num_workers=args.num_workers,
                                     prefetch_factor=args.prefetch_factor,
                                     deterministic=args.deterministic,
                                     seed=int(args.seed),
                                     empty_cache=args.empty_cache,
                                     debug_flag=args.debug_flag,
                                     comet_ml_params=comet_ml_params
                                     )

    # Call Speech Enhancement model
    speech_enhancer()


if __name__ == "__main__":
    main()
