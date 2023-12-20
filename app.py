"""
File:
    app.py

Description:
    Web application.
"""


import os
from pathlib import Path
import yaml
import torch
import torchaudio
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchinfo import summary
import gradio as gr
from src.model.SurrogateGradient import SuperSpike, SigmoidDerivative, ATan, PiecewiseLinear, SpikeFunc
from src.model.SpikingModel import UNetSNN
from src.data.constants import DataDirectories, AudioParameters
from src.data.DatasetManager import DatasetManager
from src.stft.constants import StftParameters


experiment_filename = 'SpeechEnhancement_Train_UNetSNN_2023_05_26-03_26_01_PM_37652140'
experiment_files_dir = os.path.join(Path(__file__).parent,
                                    DataDirectories.experiments_dirname,
                                    experiment_filename)


def load_params(experiment_files_dir):
    params = {}

    params_dir = os.path.join(experiment_files_dir, 'params.json')
    if os.path.exists(params_dir):
        params = yaml.safe_load(open(params_dir, 'rt'))['hyperparameters']

    for key, value in params.items():
        if value == 'True':
            params[key] = True
        elif value == 'False':
            params[key] = False

    if params['spike_fn'] == 'SuperSpike':
        spike_fn = SuperSpike
    elif params['spike_fn'] == 'SigmoidDerivative':
        spike_fn = SigmoidDerivative
    elif params['spike_fn'] == 'ATan':
        spike_fn = ATan
    elif params['spike_fn'] == 'PiecewiseLinear':
        spike_fn = PiecewiseLinear
    elif params['spike_fn'] == 'SpikeFunc':
        spike_fn = SpikeFunc

    spike_fn.spiking_mode = params['spiking_mode']
    if params['surrogate_scale'] is not None:
        spike_fn.surrogate_scale = params['surrogate_scale']

    params['spike_fn'] = spike_fn

    params['truncated_bptt_ratio'] = int(params['truncated_bptt_ratio'])

    params['debug_flag'] = False

    return params


params = load_params(experiment_files_dir)

dtype = torch.float32
device = params['device']

data_files_dir = os.path.join(DataDirectories.project_dir,
                              f'{params["task_name"]}_{DataDirectories.data_dirname}')

dataset_manager_test = DatasetManager(data_files_dir=data_files_dir,
                                      data_load=DataDirectories.data_load_test,
                                      experiment_files_dir=experiment_files_dir,
                                      plots_dir=os.path.join(experiment_files_dir, 'plots'),
                                      dtype=dtype,
                                      representation_name=params['representation_name'],
                                      representation_dir_name=params['representation_dir_name'],
                                      transform_name=params['transform_name'],
                                      debug_flag=params['debug_flag'])

batch_size = 8
dataloader_manager_test = DataLoader(dataset_manager_test, batch_size=batch_size,
                                     shuffle=False, num_workers=0,
                                     pin_memory=True, drop_last=False,
                                     sampler=None, prefetch_factor=None)


model_file_dir = os.path.join(experiment_files_dir, f'{params["task_name"]}_{params["model_name"]}_InpDim={params["input_dim"]}.pt')


def load_model(model_file_dir, verbose=False):
    net = UNetSNN(input_dim=params['input_dim'], hidden_channels_list=params.get('hidden_channels_list'),
                  output_dim=params['output_dim'], kernel_size=params.get('kernel_size'), stride=params.get('stride'),
                  padding=params.get('padding'), dilation=params.get('dilation'), bias=params['bias'],
                  padding_mode=params['padding_mode'], pooling_flag=params['pooling_flag'],
                  pooling_type=params['pooling_type'], use_same_layer=params['use_same_layer'],
                  nb_steps=params['nb_steps_bin'], truncated_bptt_ratio=params['truncated_bptt_ratio'],
                  spike_fn=params['spike_fn'], neuron_model=params['neuron_model'],
                  neuron_parameters=params['neuron_parameters'], weight_init=params['weight_init'],
                  upsample_mode=params['upsample_mode'], scale_flag=params['scale_flag'],
                  scale_factor=params['scale_factor'], bn_flag=params['bn_flag'], dropout_flag=params['dropout_flag'],
                  dropout_p=params.get('dropout_p'), device=device, dtype=dtype,
                  skip_connection_type=params['skip_connection_type'],
                  use_intermediate_output=params['use_intermediate_output']).to(device)


    checkpoint = torch.load(model_file_dir)

    state_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k =  k[len('module.'):]
        new_state_dict[k] = v

    net.load_state_dict(new_state_dict)

    if verbose:
        net.init_state(1)
        net.eval()

        print(
            summary(net, input_size=(1, 1, params['input_dim'], params['nb_steps_bin']), depth=4,
                    col_names=['kernel_size', 'output_size', 'num_params', 'mult_adds'],
                    row_settings=['var_names'], verbose=0, device=device, cache_forward_pass=True)
        )

    return net


net = load_model(model_file_dir)


n_fft = StftParameters.n_fft
win_length = StftParameters.win_length
hop_length = StftParameters.hop_length
power = StftParameters.power
normalized = StftParameters.normalized
center = StftParameters.center


def stft_splitter(x, compute_stft: bool = False):
    if compute_stft:
        x = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=power,
                                              normalized=normalized, center=center)(x.cpu())
    return torch.abs(x)[:, :, :-1, :], torch.angle(x)[:, :, :-1, :]


def stft_mixer(x_abs, x_arg):
    x = torch.complex(x_abs * torch.cos(x_arg), x_abs * torch.sin(x_arg))
    x = torch.cat((x, x[:, :, -1:, :]), 2)
    return torchaudio.transforms.InverseSpectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                                    normalized=normalized, center=center)(x.cpu())


noisy_dir = dataset_manager_test.noisyspeech_dir
clean_dir = dataset_manager_test.cleanspeech_dir
audio_filename = sorted(os.listdir(dataset_manager_test.noisyspeech_dir))

index_ = 5
noisy_audio_dir = os.path.join(noisy_dir, audio_filename[index_])


def get_audio_len(audio_dir):
    waveform, sr = torchaudio.load(audio_dir)
    if sr != AudioParameters.sample_rate:
        waveform = torchaudio.transforms.Resample(sr, AudioParameters.sample_rate).to(waveform.device)(waveform)
    len_audio = waveform.shape[1]
    return len_audio


def enhance_audio(noisy_audio_dir):
    len_audio = get_audio_len(noisy_audio_dir)

    noisy = dataset_manager_test.load_audio(noisy_audio_dir, update_info=False)
    noisy = torch.unsqueeze(noisy, dim=0)

    noisy_abs, noisy_arg = stft_splitter(noisy, True)
    noisy_abs_ = dataset_manager_test.transform_manager(noisy_abs.cpu()).to(device)
    net.init_state(noisy_abs_.shape[0])
    net.init_rec()
    cleaned_abs_, _ = net(noisy_abs_)
    cleaned_abs = dataset_manager_test.transform_manager(cleaned_abs_.cpu(), mode='inverse_transform').to(device)

    # noisy = stft_mixer(noisy_abs, noisy_arg)
    clean_rec = stft_mixer(cleaned_abs, noisy_arg.to(device)).detach()

    # noisy = noisy.view(noisy.shape[0], noisy.shape[-1])
    clean_rec = clean_rec.view(clean_rec.shape[0], clean_rec.shape[-1])
    clean_rec = clean_rec[0].cpu().numpy()

    return AudioParameters.sample_rate, clean_rec[:len_audio]


# sample_rate, clean_rec = enhance_audio(noisy_audio_dir)


def main():

    demo = gr.Interface(fn=enhance_audio,
                        inputs=gr.Audio(type='filepath'),
                        outputs=gr.Audio(),
                        examples=os.path.join(os.getcwd(), "examples"),
                        )

    demo.launch(show_api=False)


if __name__ == "__main__":
    main()
