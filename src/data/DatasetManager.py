"""
File:
    model/DatasetManager.py

Description:
    Defines the DatasetManager class.
"""


try:
    from src.lca.LcaManager import LcaManager
    from src.lca.constants import FilterBankParameters, LcaParameters
except ImportError:
    pass


import shutil
import os
from typing import Optional, Tuple
import json
import librosa
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from comet_ml import Experiment
from src.stft.StftManager import StftManager
from src.stft.constants import StftParameters
from src.visualization.VisualizationManager import VisualizationManager
from src.data.TransformManager import TransformManager
from src.data.constants import DataDirectories, AudioParameters, RepresentationName


class DatasetManager(Dataset):
    """Class that handles the creation of the training, validation and testing datasets.
    """

    def __init__(self,
                 data_files_dir: str,
                 data_load: str,
                 experiment_files_dir: str,
                 plots_dir: str,
                 dtype: torch.dtype,
                 representation_name: str,
                 representation_dir_name: str,
                 transform_name: list,
                 compute_representation: bool = False,
                 reconstruct_flag: bool = False,
                 use_ddp: bool = False,
                 experiment: Optional[Experiment] = None,
                 debug_flag: bool = False,
                 ) -> None:

        self.use_ddp = use_ddp
        self.experiment = experiment
        if self.use_ddp:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        self.device = torch.device(f'cuda:{self.rank}')
        self.dtype = dtype
        self.visualization_manager = VisualizationManager()
        self.plots_dir = plots_dir
        self.reconstruct_flag = reconstruct_flag

        self.representation_name = representation_name

        self.audio_info = None
        self._init_audio_info()

        # Set data directories
        self.audio_dir = os.path.join(data_files_dir, DataDirectories.audio_dirname)
        self.audio_info_dir = os.path.join(self.audio_dir, DataDirectories.audio_info_dirname)

        if compute_representation:
            representation_dir_name = f'{representation_name.upper()}_{AudioParameters.max_duration}s'
            if representation_name == RepresentationName.lca:
                representation_dir_name += f'_NumChannels={FilterBankParameters.num_channels}'
            elif representation_name == RepresentationName.stft:
                representation_dir_name += f'_nfft={StftParameters.n_fft}' \
                                           f'_wl={StftParameters.win_length}' \
                                           f'_hl={StftParameters.hop_length}'

        # __DEBUG__
        if debug_flag:
            DEBUGstr = '__DEBUG__'
            representation_dir_name = f'{DEBUGstr}{representation_dir_name}'
            self.audio_dir = os.path.join(data_files_dir, f'{DEBUGstr}{DataDirectories.audio_dirname}')
            self.audio_info_dir = os.path.join(self.audio_dir, f'{DataDirectories.audio_info_dirname}')

        if data_load == DataDirectories.data_load_train:
            print(f'\t - audio_dir : {self.audio_dir}')
            print(f'\t - representation_dir_name : {representation_dir_name}')

        self.representation_dir = os.path.join(data_files_dir, representation_dir_name)
        self.representation_reconstruct_dir = os.path.join(self.representation_dir, f'reconstruction')
        self.weights_dir = None
        if representation_name == RepresentationName.lca:
            self.weights_dir = os.path.join(self.representation_dir, DataDirectories.lca_weights_dirname)

        self.transform_info_dir = os.path.join(self.representation_dir, DataDirectories.transform_info_dirname)

        self.coefficients_dir = os.path.join(self.representation_dir, DataDirectories.coefficients_dirname)

        self.enhanced_coefficients_dir = os.path.join(experiment_files_dir, f'{DataDirectories.enhancedspeech_dirname}_{DataDirectories.coefficients_dirname}')
        self.enhancedspeech_dir = os.path.join(experiment_files_dir, f'{DataDirectories.enhancedspeech_dirname}_{DataDirectories.audio_dirname}')

        # Set compute_representation attribute
        self.compute_representation = compute_representation

        self.data_load = data_load  # 'train' or 'valid' or 'test'

        self.audio_type = AudioParameters.audio_type

        # Set data file names
        if not reconstruct_flag:
            self.noisyspeech_dirname = DataDirectories.noisyspeech_dirname
        else:
            self.noisyspeech_dirname = DataDirectories.cleanspeech_dirname
        self.cleanspeech_dirname = DataDirectories.cleanspeech_dirname
        self.coefficients_filename = DataDirectories.coefficients_filename

        # Set noisy and clean speech signals paths
        self.noisyspeech_dir = os.path.join(self.audio_dir, f'{self.noisyspeech_dirname}_{self.data_load}')
        self.cleanspeech_dir = os.path.join(self.audio_dir, f'{self.cleanspeech_dirname}_{self.data_load}')

        # Set coefficients file path
        self.tensor_weights_dir = None
        if representation_name == RepresentationName.lca:
            self.tensor_weights_dir = os.path.join(self.weights_dir, f'{DataDirectories.lca_weights_filename}.pt')

        self.tensor_noisyspeech_transform_info_dir = os.path.join(self.transform_info_dir,
                                                                  f'{self.noisyspeech_dirname}_{DataDirectories.transform_info_filename}.pt')
        self.tensor_cleanspeech_transform_info_dir = os.path.join(self.transform_info_dir,
                                                           f'{self.cleanspeech_dirname}_{DataDirectories.transform_info_filename}.pt')

        self.tensor_noisyspeech_coefficients_dir = os.path.join(self.coefficients_dir,
                                                               f'{self.noisyspeech_dirname}_{self.data_load}')
        self.tensor_cleanspeech_coefficients_dir = os.path.join(self.coefficients_dir,
                                                               f'{self.cleanspeech_dirname}_{self.data_load}')
        self.tensor_enhanced_coefficients_dir = os.path.join(self.enhanced_coefficients_dir)

        # check if data directories exist
        if self.rank == 0:
            if data_load == DataDirectories.data_load_train and not os.path.exists(os.path.join(self.audio_dir,
                                             f'{self.noisyspeech_dirname}_{DataDirectories.data_load_valid}')):
                self.prepare_valid_audio()
            self._check_paths()
        if self.use_ddp:
            dist.barrier()

        self.nb_steps = None
        if representation_name == RepresentationName.lca:
            self.nb_steps = int(
                ((AudioParameters.max_duration * AudioParameters.sample_rate) - FilterBankParameters.ker_len) / LcaParameters.stride) + 1
        elif representation_name == RepresentationName.stft:
            self.nb_steps = int(
                ((AudioParameters.max_duration * AudioParameters.sample_rate)) / StftParameters.hop_length) + 1

        # Set audio array parameters
        self.sample_rate = AudioParameters.sample_rate
        self.nb_samples = int(self.sample_rate * AudioParameters.max_duration)
        if representation_name == RepresentationName.lca:
            self.nb_samples = int(self.nb_steps - 1) * LcaParameters.stride + FilterBankParameters.ker_len

        if self.compute_representation:

            if self.rank == 0:
                print(f'Compute {representation_name} representation {self.data_load} data...')

                # Compute representation
                if not reconstruct_flag:
                    print(f'\t - Compute {representation_name} representation of noisy data')
                    self.compute_representation_fn(dataset_type=DataDirectories.noisyspeech_dirname,
                                                   dataset_dir=self.noisyspeech_dir,
                                                   tensor_coefficients_dir=self.tensor_noisyspeech_coefficients_dir)

                print(f'\t - Compute {representation_name} representation of clean data')
                self.compute_representation_fn(dataset_type=DataDirectories.cleanspeech_dirname,
                                               dataset_dir=self.cleanspeech_dir,
                                               tensor_coefficients_dir=self.tensor_cleanspeech_coefficients_dir)

            if self.use_ddp:
                dist.barrier()

        self.transform_manager = None
        if transform_name is not None:
            self.transform_manager = TransformManager(representation_name=representation_name,
                                                      transform_name=transform_name,
                                                      tensor_noisyspeech_transform_info_dir=
                                                      self.tensor_noisyspeech_transform_info_dir,
                                                      tensor_cleanspeech_transform_info_dir=
                                                      self.tensor_cleanspeech_transform_info_dir,
                                                      )

            if self.experiment:
                self.experiment.log_other(f'{self.representation_name}_noisy_metadata_dict',
                                          self.transform_manager.noisyspeech_transform_info)

                self.experiment.log_other(f'{self.representation_name}_clean_metadata_dict',
                                          self.transform_manager.cleanspeech_transform_info)

    def __len__(self) -> int:
        """Method that returns the number of samples.
        """
        return len(os.listdir(self.tensor_noisyspeech_coefficients_dir))

    def __getitem__(self, index: int, map_location: Optional[str] = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Method that check if a particular directory exist or not.

        Parameters
        ----------
        index: int
            Sample index.
        map_location: Optional[str]
            Remap storage location.
        """
        # Noisy input
        noisyspeech_coefficients_dir = os.path.join(self.tensor_noisyspeech_coefficients_dir,
                                                    self.coefficients_filename + str(index) + '.pt')
        noisyspeech_coefficients = torch.load(noisyspeech_coefficients_dir, map_location=map_location).detach().cpu()
        # Clean input
        cleanspeech_coefficients_dir = os.path.join(self.tensor_cleanspeech_coefficients_dir,
                                                           self.coefficients_filename + str(index) + '.pt')
        cleanspeech_coefficients = torch.load(cleanspeech_coefficients_dir, map_location=map_location).detach().cpu()
        return noisyspeech_coefficients, cleanspeech_coefficients, index

    def _init_audio_info(self) -> None:
        """Method that defines audio info dictionary.
        """

        self.audio_info = {'nb_short_audio': 0,
                           'audio_duration': 0.,
                           'max_duration': 0.,
                           'min_duration': 100.,
                           }

    def _save_audio_info(self, file_name: str) -> None:
        """Method that saves audio info dictionary.

        Parameters
        ----------
        file_name: str
            json file name.
        """

        file_rec_dir = os.path.join(self.audio_info_dir, f'{file_name}.json')
        file_rec_dict_file = json.dumps(self.audio_info, indent=4)
        with open(file_rec_dir, 'w') as outfile:
            outfile.write(file_rec_dict_file)

    def prepare_valid_audio(self) -> None:
        """Method that checks the existance of 'valid' audio data.
        """

        # indices of the training files to move
        nb_train_data = len(os.listdir(self.noisyspeech_dir))
        nb_valid_data = len(os.listdir(os.path.join(self.audio_dir,
                                             f'{self.noisyspeech_dirname}_{DataDirectories.data_load_test}')))
        valid_data_index = np.random.choice(nb_train_data, nb_valid_data, replace=False)

        # Init directories
        valid_noisyspeech_dir = os.path.join(self.audio_dir,
                                             f'{self.noisyspeech_dirname}_{DataDirectories.data_load_valid}')
        valid_cleanspeech_dir = os.path.join(self.audio_dir,
                                             f'{self.cleanspeech_dirname}_{DataDirectories.data_load_valid}')
        valid_speech_dir_list = [valid_noisyspeech_dir, valid_cleanspeech_dir]
        train_speech_dir_list = [self.noisyspeech_dir, self.cleanspeech_dir]

        for i in range(len(valid_speech_dir_list)):
            valid_speech_dir_i = valid_speech_dir_list[i]
            train_speech_dir_i = train_speech_dir_list[i]

            # create valid dir
            os.makedirs(valid_speech_dir_i)

            # List of all the files in the train folder
            train_files = os.listdir(train_speech_dir_list[i])

            # iterate over the indices and move the corresponding file to the valid folder
            for index in valid_data_index:
                file_to_move = train_files[index]
                shutil.move(os.path.join(train_speech_dir_i, file_to_move),
                            os.path.join(valid_speech_dir_i, file_to_move))

    def _check_paths(self) -> None:
        """Method that verifies the existance of all necessary file paths.
        """

        # check if data paths exist
        assert os.path.exists(self.audio_dir), f'Audio path "{self.audio_dir}" is required'

        if not os.path.exists(self.audio_info_dir):
            os.makedirs(self.audio_info_dir)

        if not os.path.exists(self.representation_dir):
            if not self.compute_representation:
                assert False, f'representation data path "{self.representation_dir}" is required'
            else:
                os.makedirs(self.representation_dir)

        if self.representation_name == RepresentationName.lca:
            if not os.path.exists(self.weights_dir):
                if not self.compute_representation:
                    assert False, f'LCA weights path "{self.weights_dir}" is required'
                else:
                    os.makedirs(self.weights_dir)

        if not os.path.exists(self.transform_info_dir):
            if not self.compute_representation and self.transform_manager is not None:
                assert False, f'Transform metadata path "{self.transform_info_dir}" is required'
            else:
                os.makedirs(self.transform_info_dir)

        if not os.path.exists(self.representation_reconstruct_dir):
            os.makedirs(self.representation_reconstruct_dir)

        if not os.path.exists(self.coefficients_dir):
            if not self.compute_representation:
                # assert False, f'{self.representation_name} representation path is required'
                pass
            else:
                os.makedirs(self.coefficients_dir)

        if not os.path.exists(self.tensor_noisyspeech_coefficients_dir):
            os.makedirs(self.tensor_noisyspeech_coefficients_dir)

        if not os.path.exists(self.tensor_cleanspeech_coefficients_dir):
            os.makedirs(self.tensor_cleanspeech_coefficients_dir)

        if not os.path.exists(self.enhancedspeech_dir):
            os.makedirs(self.enhancedspeech_dir)

        if not os.path.exists(self.enhanced_coefficients_dir):
            os.makedirs(self.enhanced_coefficients_dir)

        if not os.path.exists(self.tensor_enhanced_coefficients_dir):
            os.makedirs(self.tensor_enhanced_coefficients_dir)

    def get_weights(self, verbose: bool = False) -> torch.Tensor:
        """Method that loads LCA weights.

        Parameters
        ----------
        verbose: bool
            Boolean that indicates weather to print specific output.
        """

        # Load npy array
        tensor_weights = torch.load(self.tensor_weights_dir, map_location=self.device)

        if verbose:
            print('\t - Weights array shape')
            print(f'\t\t - weights [nb_channels, 1, kernel_length] : {tensor_weights.shape}')

        return tensor_weights.to(dtype=self.dtype)

    def get_coefficients(self, coefficients_dir: str, index: int, map_location:Optional[str] = None,
                         verbose: bool = False) -> torch.Tensor:
        """Method that loads representation coefficients.

        Parameters
        ----------
        coefficients_dir: str
            Data directory.
        index: int
            Sample index.
        map_location: Optional[str]
            Remap storage location.
        verbose: bool
            Boolean that indicates weather to print specific output.
        """

        # Load npy array
        str_index = str(index)
        if map_location is None:
            map_location = self.device
        coefficients_array = torch.load(os.path.join(coefficients_dir, self.coefficients_filename + str_index + '.pt'),
                                        map_location=map_location)

        if verbose:
            print('\t - Coefficients array shape')
            print(f'\t\t   [nb_channels, time_steps] : {coefficients_array.shape}')

        return coefficients_array

    def compute_representation_fn(self, dataset_type: str, dataset_dir: str, tensor_coefficients_dir: str):
        """Method that computes data representation using audio data.

        Parameters
        ----------
        dataset_type: str
            'noisy' or 'clean' audio.
        dataset_dir: str
            Audio data directory.
        tensor_coefficients_dir: str
            Representation coefficients directory.
        """

        print(f'\t - Loading {dataset_type.lower()} audio...')
        # Load audio file and convert to tensor
        dataset_tensor = self.audio_to_tensor(audio_dir=dataset_dir)

        # Save audio info
        self._save_audio_info(file_name=f'{dataset_type}_{self.data_load}')
        self._init_audio_info()

        tensor_transform_info_dir = None
        if dataset_type == self.noisyspeech_dirname:
            tensor_transform_info_dir = self.tensor_noisyspeech_transform_info_dir
        elif dataset_type == self.cleanspeech_dirname:
            tensor_transform_info_dir = self.tensor_cleanspeech_transform_info_dir

        representation_manager = None
        if self.representation_name == RepresentationName.lca:
            print(f'\t - Create LCA manager of {dataset_type.lower()} data...')
            # Create LCA manager instance
            representation_manager = LcaManager(dataset_tensor=dataset_tensor,
                                     dataset_dir=dataset_dir,
                                     dataset_type=dataset_type,
                                     data_load=self.data_load,
                                     tensor_weights_dir=self.tensor_weights_dir,
                                     tensor_coefficients_dir=tensor_coefficients_dir,
                                     tensor_transform_info_dir=tensor_transform_info_dir,
                                     lca_reconstruct_dir=self.representation_reconstruct_dir)

            print(f'\t - Compute {self.representation_name} representation of {dataset_type.lower()} data...')

            # Compute LCA coefficients
            representation_manager()

        elif self.representation_name == RepresentationName.stft:
            print(f'\t - Create STFT manager of {dataset_type.lower()} data...')
            # Create stft manager instance
            representation_manager = StftManager(dataset_tensor=dataset_tensor,
                                       dataset_dir=dataset_dir,
                                       dataset_type=dataset_type,
                                       data_load=self.data_load,
                                       tensor_coefficients_dir=tensor_coefficients_dir,
                                       tensor_transform_info_dir=tensor_transform_info_dir,
                                       stft_reconstruct_dir=self.representation_reconstruct_dir)

            print(f'\t - Compute STFT representation of {dataset_type.lower()} data...')

            # Compute stft coefficients
            representation_manager()

        self.experiment.log_other(f'{self.representation_name}_params_dict', representation_manager.params_dict)
        # self.experiment.log_other(f'{self.representation_name}__{dataset_type}__{self.data_load}__metadata_dict',
        #                           representation_manager.transform_info)

    def audio_to_tensor(self, audio_dir: str, verbose: bool = True) -> torch.Tensor:
        """Method that loads audio data into one tensor.

        Parameters
        ----------
        audio_dir: str
            Audio data directory.
        verbose: bool
            Boolean that indicates weather to print specific output.
        """

        audio_list = sorted(os.listdir(audio_dir))
        nb_audio = len(audio_list)
        self.audio_info['nb_audio'] = nb_audio
        waveform_tensor = torch.zeros((nb_audio, 1, self.nb_samples), dtype=self.dtype)

        for i in tqdm(range(nb_audio), leave=True, desc='\t Audio Loading'):
            # load the audio file
            file_dir = os.path.join(audio_dir, audio_list[i])
            waveform_tensor_i = self.load_audio(file_dir=file_dir)
            waveform_tensor[i] = waveform_tensor_i.clone()

        if verbose:
            print(f'\t\t - Number of audio: {self.audio_info["nb_audio"]}')
            print(f'\t\t - Number of short audio: {self.audio_info["nb_short_audio"]}')
            print(f'\t\t - Audio duration: {self.audio_info["audio_duration"]/3600:.2f} hours')
            print(f'\t\t - Maximum audio duration: {self.audio_info["max_duration"]:.2f} seconds')
            print(f'\t\t - Minimum audio duration: {self.audio_info["min_duration"]:.2f} seconds')

        return waveform_tensor

    def load_audio(self, file_dir: str, normalize: bool = True, pad_flag: bool = True, pad_mode: str = 'circular',
                   update_info: bool = True) -> torch.Tensor:
        """Method that loads audio waveform.

        Parameters
        ----------
        file_dir: str
            Audio file directory.
        normalize: bool
            Boolean that indicates weather to normalize audio data.
        pad_flag: bool
            Boolean that indicates weather to pad audio waveform.
        pad_mode: str
            Audio padding mode: 'constant' using zeros or 'circular'.
        update_info: bool
            Boolean that indicates weather to update metadata dictionary.
        """

        # Load signal
        waveform, sr = torchaudio.load(file_dir)

        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate).to(waveform.device)(waveform)

        if normalize:
            waveform = torch.tensor(librosa.util.normalize(waveform.numpy(), axis=1))

        waveform_length = waveform.shape[-1]

        if update_info:
            self.audio_info['audio_duration'] += waveform_length / self.sample_rate
            self.audio_info['max_duration'] = max(self.audio_info['max_duration'], waveform_length / self.sample_rate)
            self.audio_info['min_duration'] = min(self.audio_info['min_duration'], waveform_length / self.sample_rate)

        if pad_flag:
            if waveform_length < self.nb_samples:
                pad_length = self.nb_samples - waveform_length
                if pad_mode == 'constant':
                    waveform = F.pad(waveform, pad=(0, pad_length), mode='constant', value=0)
                elif pad_mode == 'circular':
                    while waveform.shape[-1] < self.nb_samples:
                        waveform = torch.cat((waveform, waveform), -1)
                else:
                    valid_pad_mode = ['constant', 'circular']
                    assert pad_mode in valid_pad_mode, f'pad_mode parameter "{pad_mode}" should be in {valid_pad_mode}'

                if update_info:
                    self.audio_info['nb_short_audio'] += 1

            waveform = waveform[:, :self.nb_samples]

        return waveform.view(1, -1)

    def audio_reconstruction(self, normalize: bool = True, map_location: Optional[str] = 'cpu') -> None:
        """Method that reconstructs audio signals using enhanced coefficients.

        Parameters
        ----------
        normalize: bool
            Boolean that indicates weather to normalize audio data.
        map_location: Optional[str]
            Remap storage location.
        """

        # Load noisy audio list
        list_noisyspeech_audio = sorted(os.listdir(self.noisyspeech_dir))

        # Load enhanced coefficients
        list_enhanced_coefficients = sorted(os.listdir(self.enhanced_coefficients_dir),
                                            key=lambda x: int(x[len(self.coefficients_filename):-3]))

        weight = None
        if self.representation_name == RepresentationName.lca:
            weight = self.get_weights()

        progress_bar = tqdm(total=len(list_enhanced_coefficients), leave=False, desc='Audio reconstruction')

        for i, file in enumerate(list_enhanced_coefficients):

            progress_bar.update()

            file_dir = os.path.join(self.enhanced_coefficients_dir, file)
            enhanced_coefficients_i = torch.load(file_dir, map_location=map_location)

            # Reconstruct audio
            if self.representation_name == RepresentationName.lca:
                enhanced_speech_i = F.conv_transpose1d(
                    input=enhanced_coefficients_i.view(
                        (-1, enhanced_coefficients_i.shape[-2], enhanced_coefficients_i.shape[-1])),
                    weight=weight.cpu(),
                    stride=LcaParameters.stride)
            elif self.representation_name == RepresentationName.stft:
                enhanced_speech_i = torchaudio.transforms.InverseSpectrogram(n_fft=StftParameters.n_fft,
                                                                             win_length=StftParameters.win_length,
                                                                             hop_length=StftParameters.hop_length,
                                                                             normalized=StftParameters.normalized,
                                                                             center=StftParameters.center,
                                                                             )(enhanced_coefficients_i)

            enhancedspeech_dir_i = os.path.join(self.enhancedspeech_dir,
                                                f'{DataDirectories.enhancedspeech_filename}_{list_noisyspeech_audio[i]}')

            if normalize:
                enhanced_speech_i = torch.tensor(librosa.util.normalize(enhanced_speech_i.view(1, -1).numpy(), axis=-1))

            # Get audio length
            noisyspeech_dir_i = os.path.join(self.noisyspeech_dir, f'{list_noisyspeech_audio[i]}')
            waveform = self.load_audio(file_dir=noisyspeech_dir_i, pad_flag=False, update_info=False)

            # Save reconstructed audio files
            torchaudio.save(enhancedspeech_dir_i, enhanced_speech_i[:waveform.shape[-1]], self.sample_rate, encoding='PCM_F')

