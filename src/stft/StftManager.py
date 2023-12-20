"""
File:
    stft/StftManager.py

Description:
    Defines the StftManager class.
"""

import os
from pathlib import Path
import numpy as np
import json
from src.stft.Stft import Stft
from src.stft.constants import StftParameters
from src.data.constants import DataDirectories
import torch

class StftManager:
    """Class that implements the STFT manager.
    """

    def __init__(self,
                 dataset_tensor: torch.Tensor,
                 dataset_dir: str,
                 dataset_type: str,
                 data_load: str,
                 tensor_coefficients_dir: str,
                 tensor_transform_info_dir: str,
                 stft_reconstruct_dir: str):

        # Define STFT
        self.stft = Stft(n_fft=StftParameters.n_fft,
                         win_length=StftParameters.win_length,
                         hop_length=StftParameters.hop_length,
                         power=StftParameters.power,
                         normalized=StftParameters.normalized,
                         center=StftParameters.center,
                         batch_size=StftParameters.batch_size,
                         stft_reconstruct_dir=stft_reconstruct_dir
                         )

        self.params_dict = None

        self.stft_reconstruct_dir = stft_reconstruct_dir

        # Set data instances
        self.dataset_tensor = dataset_tensor
        self.dataset_dir = sorted(os.listdir(dataset_dir))
        self.dataset_type = dataset_type
        self.data_load = data_load

        # Set saving instances
        self.tensor_coefficients_dir = tensor_coefficients_dir
        self.tensor_transform_info_dir = tensor_transform_info_dir

        # Init global statistics
        self.transform_info = {'mean': [],
                               'std': [],
                               'max': [],
                               'min': [],
                               'maxabs': [],
                               'quantile_max': [],
                               'quantile_min': [],
                               'quantile_maxabs': [],
                               }


    def compute_stft(self, verbose: bool = True):
        """Method that handles STFT computation.

        Parameters
        ----------
        verbose: bool
            Boolean that indicates weather to print specific output.
        """
        batch_list = torch.split(self.dataset_tensor, StftParameters.batch_size, dim=0)
        batch_dir_list = [self.dataset_dir[i:i + StftParameters.batch_size] for i in
                          range(0, len(self.dataset_dir), StftParameters.batch_size)]

        mse_list = []
        snr_list = []
        nb_batch = len(batch_list)

        self.stft.dataset_type = self.dataset_type
        self.stft.data_load = self.data_load

        if verbose:
            print('\t   Number of audio batch: {}'.format(nb_batch))

        nb_saved_tensor_coefficients = 0
        for i in range(nb_batch):
            self.stft.mini_batch = batch_list[i]
            self.stft.mini_batch_dir = batch_dir_list[i]
            self.stft.stft_fn(batch_index=i)
            self.update_transform_info(self.stft.stft_coefficients)
            self.save_coefficients(self.stft.stft_coefficients, nb_saved_tensor_coefficients)
            nb_saved_tensor_coefficients += len(batch_dir_list[i])
            mse_list.append(self.stft.mse.item())
            snr_list.append(self.stft.snr.item())

        return np.mean(mse_list), np.mean(snr_list)

    def _dict_to_tensor(self):
        """Method that transformers dictionary values to tensors.
        """
        for k, v in self.transform_info.items():
            self.transform_info[k] = torch.tensor(v)

    def update_transform_info(self, coefficients: torch.Tensor, max_qv: float = 0.99):
        """Method that handles Transforms metadata computation.

        Parameters
        ----------
        coefficients: torch.Tensor
            STFT data tensor.
        max_qv: float
            A scalar parameter in [0, 1].
        """
        for i in range(coefficients.shape[0]):
            coefficients_i = torch.abs(coefficients.clone()[i])

            self.transform_info['mean'].append(torch.mean(coefficients_i))
            self.transform_info['std'].append(torch.std(coefficients_i))
            self.transform_info['max'].append(torch.max(coefficients_i))
            self.transform_info['min'].append(torch.min(coefficients_i))
            self.transform_info['maxabs'].append(torch.max(torch.abs(coefficients_i)))
            self.transform_info['quantile_max'].append(torch.quantile(coefficients_i, max_qv))
            self.transform_info['quantile_min'].append(torch.quantile(coefficients_i, 1 - max_qv))
            self.transform_info['quantile_maxabs'].append(torch.quantile(torch.abs(coefficients_i), max_qv))

    def save_transform_info(self, verbose: bool = False):
        """Method that saves Transforms metadata.

        Parameters
        ----------
        verbose: bool
            Boolean that indicates weather to print specific output.
        """
        if verbose:
            print('\t\t - dict transform info')
            for k, v in self.transform_info.items():
                print(f'\t\t - tensor_transform_info [{k}] : {v}')

        # Save transform info
        torch.save(self.transform_info, self.tensor_transform_info_dir)

    def save_coefficients(self, coefficients: torch.Tensor, nb_saved_tensor_coefficients: int, verbose: bool = False):
        """Method that saves STFT data.

        Parameters
        ----------
        coefficients: torch.Tensor
            STFT data tensor.
        nb_saved_tensor_coefficients: int
            Number of saved data files.
        verbose: bool
            Boolean that indicates weather to print specific output.
        """
        if verbose:
            print('\t\t - tensor_coefficients_dir = {}'.format(self.tensor_coefficients_dir))
            print('\t\t   [nb_batch, nb_channels, time_steps] : {}'.format(coefficients.shape))

        # Save coefficients to disk
        batch_size = coefficients.shape[0]
        for i in range(batch_size):
            index = i + nb_saved_tensor_coefficients
            torch.save(coefficients.data[i].clone(),
                       os.path.join(self.tensor_coefficients_dir,
                                    f'{DataDirectories.coefficients_filename}{str(index)}.pt'))

    def _save_json_file(self, file_dir: str, file_name: str, file_rec_dict: dict):
        """Method that saves file_rec_dict within a json file.

        Parameters
        ----------
        file_dir: str
            File directory.
        file_name: str
            File name.
        file_rec_dict: dict
            Dictionary to  be saved to a file.
        """
        # Check path
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        # Dump the file_rec_dict dictionary into a json file
        rec_dir = os.path.join(file_dir, file_name + '.json')
        if not os.path.exists(rec_dir):
            rec_file = json.dumps(file_rec_dict, indent=4)
            with open(rec_dir, 'w') as outfile:
                outfile.write(rec_file)

    def __call__(self) -> None:

        # Save STFT parameters
        self.params_dict = {'n_fft': StftParameters.n_fft,
                            'win_length': StftParameters.win_length,
                            'hop_length': StftParameters.hop_length,
                            'power': StftParameters.power,
                            'normalized': StftParameters.normalized,
                            'center': StftParameters.center,
                            'batch_size': StftParameters.batch_size,
                            }

        self._save_json_file(file_dir=self.stft_reconstruct_dir, file_name='params', file_rec_dict=self.params_dict)

        # STFT computation
        mse, snr = self.compute_stft()
        print('\t - STFT (' + self.dataset_type + f' data): SNR = {snr:.2f} dB')

        # Save STFT results
        stft_results_dict = {'SNR (dB)': float(snr),
                             }

        self._save_json_file(file_dir=os.path.join(self.stft_reconstruct_dir, 'results'),
                             file_name=f'{self.dataset_type}_{self.data_load}',
                             file_rec_dict=stft_results_dict)

        # Save STFT transform metadata
        self._dict_to_tensor()

        if self.data_load == DataDirectories.data_load_train:
            self.save_transform_info()

        __tensor_transform_info_dir = os.path.join(Path(self.tensor_transform_info_dir).parent.absolute(),
                                                   '__transform_info__')
        if not os.path.exists(__tensor_transform_info_dir):
            os.makedirs(__tensor_transform_info_dir)
        torch.save(self.transform_info, os.path.join(__tensor_transform_info_dir,
                                                     f'{self.dataset_type}_{self.data_load}.pt'))
