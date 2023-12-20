"""
File:
    stft/Stft.py

Description:
    Defines the Stft class.
"""

import os
from typing import Optional
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from torchmetrics import SignalNoiseRatio
from src.data.constants import AudioParameters


class Stft:
    """Class that implements the STFT computation.
    """

    def __init__(self,
                 n_fft: int,
                 win_length: int,
                 hop_length: int,
                 power: Optional[float],
                 normalized: bool,
                 center: bool,
                 batch_size: int,
                 stft_reconstruct_dir: str):
        super(Stft, self).__init__()

        # define data_files type
        self.dtype = torch.float

        # Define device
        self.device = torch.device(f'cuda:{dist.get_rank()}')

        # Stft parameters
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.power = power
        self.normalized = normalized
        self.center = center
        self.batch_size = batch_size

        # Audio reconstruction path
        self.stft_reconstruct_dir = stft_reconstruct_dir

        # mini_batch
        self.__mini_batch = None
        self.__mini_batch_dir = None

        # Final STFT outputs
        self.stft_coefficients = None
        self.dataset_type = None
        self.data_load = None
        self.rec_audio = None

        # Evaluation metrics
        self.mse = None
        self.snr = None

    @property
    def mini_batch(self):
        return self.__mini_batch

    @mini_batch.setter
    def mini_batch(self, value):
        if value is not None:
            self.__mini_batch = value

    @property
    def mini_batch_dir(self):
        return self.__mini_batch_dir

    @mini_batch_dir.setter
    def mini_batch_dir(self, value):
        if value is not None:
            self.__mini_batch_dir = value

    def _audio_reconstruction(self):
        """Method that handles ISTFT computation for reconstruction quality evaluation.
        """
        self.rec_audio = torchaudio.transforms.InverseSpectrogram(n_fft=self.n_fft,
                                                                  win_length=self.win_length,
                                                                  hop_length=self.hop_length,
                                                                  normalized=self.normalized,
                                                                  center=self.center,
                                                                  )(self.stft_coefficients)

    def _compute_metrics(self):
        """Method that computes reconstruction quality evaluation metrics.
        """
        self.mse = torch.nn.MSELoss()(self.rec_audio, self.mini_batch).to(device=self.device)
        self.snr = SignalNoiseRatio()(self.rec_audio, self.mini_batch).to(device=self.device)

    def _save_rec_audio(self):
        """Method that saves reconstructed audio.
        """
        for i in range(self.rec_audio.shape[0]):

            stft_audio_dir = os.path.join(self.stft_reconstruct_dir, 'audio')
            if not os.path.exists(stft_audio_dir):
                os.makedirs(stft_audio_dir)

            dataset_type_lca_audio_dir = os.path.join(stft_audio_dir, f'{self.dataset_type}_{self.data_load}')
            if not os.path.exists(dataset_type_lca_audio_dir):
                os.makedirs(dataset_type_lca_audio_dir)

            rec_dir_i = os.path.join(dataset_type_lca_audio_dir, self.mini_batch_dir[i])
            rec_i = self.rec_audio.clone().detach().cpu()[i]
            torchaudio.save(rec_dir_i, rec_i, AudioParameters.sample_rate, encoding='PCM_F')

    def stft_fn(self, batch_index: int, verbose: bool = False):
        """ STFT computation method.
        """

        print(f'\t STFT computation (batch_index = {batch_index})')

        # Compute stft
        self.stft_coefficients = torchaudio.transforms.Spectrogram(n_fft=self.n_fft,
                                                                   win_length=self.win_length,
                                                                   hop_length=self.hop_length,
                                                                   power=self.power,
                                                                   normalized=self.normalized,
                                                                   center=self.center,
                                                                   )(self.mini_batch)

        if verbose and batch_index == 0:
            print(f'\t\t - self.mini_batch.shape = {self.mini_batch.shape}')
            print(f'\t\t - self.stft_coefficients.shape = {self.stft_coefficients.shape}')

        # Reconstruct audio
        self._audio_reconstruction()

        # Compute evaluation metrics
        self._compute_metrics()

        # Save reconstructed audio
        if self.data_load == 'test':
            self._save_rec_audio()

