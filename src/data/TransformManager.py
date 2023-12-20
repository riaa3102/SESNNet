"""
File:
    data/TransformManager.py

Description:
    Defines the TransformManager class.
"""

from typing import Optional
import torch
from src.data.constants import DataDirectories, RepresentationName


class TransformManager(object):
    """Class that handles the transformation of the input data.
    """

    def __init__(self,
                 representation_name :str,
                 transform_name: list,
                 tensor_noisyspeech_transform_info_dir: str,
                 tensor_cleanspeech_transform_info_dir: str
                 ) -> None:

        self.representation_name = representation_name

        self.tensor_noisyspeech_transform_info_dir = tensor_noisyspeech_transform_info_dir
        self.tensor_cleanspeech_transform_info_dir = tensor_cleanspeech_transform_info_dir

        self.noisyspeech_transform_info = self.get_transform_info(self.tensor_noisyspeech_transform_info_dir)
        self.cleanspeech_transform_info = self.get_transform_info(self.tensor_cleanspeech_transform_info_dir)

        self.transform_info = None
        self.transform_metadata = []

        self.transform_name = transform_name
        self.transform_fn = []
        self.inverse_transform_fn = []
        self._compose()

    @staticmethod
    def _reduce_transform_info(tensor_transform_info: dict, mean_flag: bool = True) -> dict:
        """Method that reduces transformation metadata.

        Parameters
        ----------
        tensor_transform_info: dict
            transformation metadata dictionary.
        mean_flag: bool
            Boolean that indicates weather to compute only mean of metadata.
        """

        if mean_flag:
            for k, v in tensor_transform_info.items():
                tensor_transform_info[k] = torch.mean(v)
        else:
            tensor_transform_info['mean'] = torch.mean(tensor_transform_info['mean'])
            tensor_transform_info['std'] = torch.mean(tensor_transform_info['std'])
            tensor_transform_info['max'] = torch.max(tensor_transform_info['max'])
            tensor_transform_info['min'] = torch.min(tensor_transform_info['min'])
            tensor_transform_info['maxabs'] = torch.max(tensor_transform_info['maxabs'])
            tensor_transform_info['quantile_max'] = torch.max(tensor_transform_info['quantile_max'])
            tensor_transform_info['quantile_min'] = torch.min(tensor_transform_info['quantile_min'])
            tensor_transform_info['quantile_maxabs'] = torch.max(tensor_transform_info['quantile_maxabs'])

        return tensor_transform_info

    def get_transform_info(self, tensor_transform_info_dir: str, map_location:Optional[str] = None,
                           verbose: bool = False) -> dict:
        """Method that loads transformation metadata.

        Parameters
        ----------
        tensor_transform_info_dir: dict
            transformation metadata directory.
        map_location: Optional[str]
            Remap storage location.
        verbose: bool
            Boolean that indicates weather to print specific output.
        """

        tensor_transform_info = torch.load(tensor_transform_info_dir, map_location=map_location)
        tensor_transform_info = self._reduce_transform_info(tensor_transform_info)

        if verbose:
            print(f'\t - {self.representation_name} representation transform info:')
            print(f'\t - File dir : {tensor_transform_info_dir}')
            for k, v in tensor_transform_info.items():
                print(f'\t\t - tensor_transform_info [{k}] : {v}')

        return tensor_transform_info

    def _compose(self) -> None:
        """Method that creates a list of transform/inverse transform functions.
        """

        for tsfrm in self.transform_name:
            self.transform_fn.append(getattr(self, tsfrm))
            self.inverse_transform_fn.append(getattr(self, f'inverse_{tsfrm}'))
        self.inverse_transform_fn.reverse()

    def maxabs(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Method that scales input data by its maximum absolute value.

        Parameters
        ----------
        coefficients: torch.Tensor
            Input data.
        """

        maxabs = self.transform_info['maxabs']

        coefficients_transform = coefficients.clone() / maxabs

        return coefficients_transform

    def quantile_maxabs(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Method that scales input data by its q-th quantile metadata.

        Parameters
        ----------
        coefficients: torch.Tensor
            Input data.
        """

        maxabs = self.transform_info['quantile_maxabs']

        coefficients_transform = coefficients.clone() / maxabs

        return coefficients_transform

    def normalize(self, coefficients: torch.Tensor, low: float = 0., up: float = 1., epsilon: float = 1e-8,
                  quantile_flag: bool = False) -> torch.Tensor:
        """Method that normalizes input data.

        Parameters
        ----------
        coefficients: torch.Tensor
            Input data.
        low: float
            Lower bound.
        up: float
            Upper bound.
        epsilon: float
            A small value to avoid computation error.
        quantile_flag: bool
            Boolean that indicates weather to scale data by its q-th quantile metadata.
        """

        if quantile_flag:
            max = self.transform_info['quantile_max']
            min = self.transform_info['quantile_min']
        else:
            max = self.transform_info['max']
            min = self.transform_info['min']

        coefficients_transform = None
        if self.representation_name == RepresentationName.lca:
            coefficients_transform = torch.zeros_like(coefficients)
            nonzero_index = torch.nonzero(coefficients, as_tuple=True)
            coefficients_transform[nonzero_index] = (up - low) * (
                        (coefficients[nonzero_index] - min) / (max - min + epsilon)) + low
        elif self.representation_name == RepresentationName.stft:
            coefficients_transform = (up - low) * ((coefficients - min) / (max - min + epsilon)) + low

        return coefficients_transform

    def standardize(self, coefficients: torch.Tensor, mean_: float = 0., std_: float = 1.) -> torch.Tensor:
        """Method that standardizes input data.

        Parameters
        ----------
        coefficients: torch.Tensor
            Input data.
        mean_: float
            Target mean.
        std_: float
            Target standard deviation.
        """

        mean = self.transform_info['mean']
        std = self.transform_info['std']

        coefficients_transform = None
        if self.representation_name == RepresentationName.lca:
            coefficients_transform = torch.zeros_like(coefficients)
            nonzero_index = torch.nonzero(coefficients, as_tuple=True)
            coefficients_transform[nonzero_index] = (((coefficients[nonzero_index] - mean) / std) * std_) + mean_
        elif self.representation_name == RepresentationName.stft:
            coefficients_transform = (((coefficients - mean) / std) * std_) + mean_

        return coefficients_transform

    def shift(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Method that shifts input data.

        Parameters
        ----------
        coefficients: torch.Tensor
            Input data.
        """

        min = self.transform_info['min']
        shift_v = abs(min(0., min))

        coefficients_transform = None
        if self.representation_name == RepresentationName.lca:
            coefficients_transform = torch.zeros_like(coefficients)
            nonzero_index = torch.nonzero(coefficients, as_tuple=True)
            coefficients_transform[nonzero_index] = coefficients[nonzero_index] + shift_v
        elif self.representation_name == RepresentationName.stft:
            coefficients_transform = coefficients + shift_v

        return coefficients_transform

    def log(self, coefficients: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """Method that computes logarithmic power of input data.

        Parameters
        ----------
        coefficients: torch.Tensor
            Input data.
        epsilon: float
            A small value to avoid computation error.
        """

        epsilon = torch.tensor([epsilon], dtype=coefficients.dtype, device=coefficients.device)
        coefficients_transform = None
        if self.representation_name == RepresentationName.lca:
            # log-modulus transformation
            coefficients_transform = torch.sign(coefficients) * \
                                     torch.log(torch.abs(coefficients) + 1.)
        elif self.representation_name == RepresentationName.stft:
            coefficients_transform = torch.log(torch.max(coefficients, epsilon))

        return coefficients_transform

    def log_power(self, coefficients: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """Method that computes logarithmic power of input data.

        Parameters
        ----------
        coefficients: torch.Tensor
            Input data.
        epsilon: float
            A small value to avoid computation error.
        """

        epsilon = torch.tensor([epsilon], dtype=coefficients.dtype, device=coefficients.device)
        coefficients_transform = None
        if self.representation_name == RepresentationName.lca:
            # log-modulus transformation
            coefficients_transform = torch.sign(coefficients) * \
                                     torch.log(torch.pow(torch.abs(coefficients) + 1., 2))
        elif self.representation_name == RepresentationName.stft:
            coefficients_transform = torch.log(torch.max(torch.pow(coefficients, 2), epsilon))

        return coefficients_transform

    def log10(self, coefficients: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """Method that computes loga10 of input data.

        Parameters
        ----------
        coefficients: torch.Tensor
            Input data.
        epsilon: float
            A small value to avoid computation error.
        """

        epsilon = torch.tensor([epsilon], dtype=coefficients.dtype, device=coefficients.device)
        coefficients_transform = None
        if self.representation_name == RepresentationName.lca:
            coefficients_transform = torch.sign(coefficients) * torch.log10(torch.abs(coefficients) + 1.)
        elif self.representation_name == RepresentationName.stft:
            coefficients_transform = torch.log10(torch.max(coefficients, epsilon))

        return coefficients_transform

    def inverse_maxabs(self, coefficients_transform: torch.Tensor) -> torch.Tensor:
        maxabs = self.transform_info['maxabs']

        coefficients = coefficients_transform.clone() * maxabs

        return coefficients

    def inverse_quantile_maxabs(self, coefficients_transform: torch.Tensor) -> torch.Tensor:
        maxabs = self.transform_info['quantile_maxabs']

        coefficients = coefficients_transform.clone() * maxabs

        return coefficients

    def inverse_normalize(self, coefficients_transform: torch.Tensor, low: float = 0., up: float = 1.,
                          epsilon: float = 1e-8, quantile_flag: bool = False) -> torch.Tensor:
        if quantile_flag:
            max = self.transform_info['quantile_max']
            min = self.transform_info['quantile_min']
        else:
            max = self.transform_info['max']
            min = self.transform_info['min']

        coefficients = None
        if self.representation_name == RepresentationName.lca:
            coefficients = torch.zeros_like(coefficients_transform)
            nonzero_index = torch.nonzero(coefficients_transform, as_tuple=True)
            coefficients[nonzero_index] = (coefficients_transform[nonzero_index] * (max - min + epsilon) + min) / (
                        up - low) - low
        elif self.representation_name == RepresentationName.stft:
            coefficients =  (coefficients_transform * (max - min + epsilon) + min) / (up - low) - low

        return coefficients

    def inverse_standardize(self, coefficients_transform: torch.Tensor, mean_: float = 0., std_: float = 1.) \
            -> torch.Tensor:
        mean = self.transform_info['mean']
        std = self.transform_info['std']

        coefficients = None
        if self.representation_name == RepresentationName.lca:
            coefficients = torch.zeros_like(coefficients_transform)
            nonzero_index = torch.nonzero(coefficients_transform, as_tuple=True)
            coefficients[nonzero_index] = (((coefficients_transform[nonzero_index] - mean_) / std_) * std) + mean
        elif self.representation_name == RepresentationName.stft:
            coefficients = (((coefficients_transform - mean_) / std_) * std) + mean

        return coefficients

    def inverse_shift(self, coefficients_transform: torch.Tensor) -> torch.Tensor:
        min = self.transform_info['min']
        shift_v = abs(min(0., min))

        coefficients = None
        if self.representation_name == RepresentationName.lca:
            coefficients = torch.zeros_like(coefficients_transform)
            nonzero_index = torch.nonzero(coefficients_transform, as_tuple=True)
            coefficients[nonzero_index] = coefficients_transform[nonzero_index] + shift_v
        elif self.representation_name == RepresentationName.stft:
            coefficients = coefficients_transform - shift_v

        return coefficients

    def inverse_log(self, coefficients_transform: torch.Tensor) -> torch.Tensor:

        coefficients = None
        if self.representation_name == RepresentationName.lca:
            coefficients = torch.sign(coefficients_transform) * \
                           (torch.exp(torch.abs(coefficients_transform)) - 1.)
        elif self.representation_name == RepresentationName.stft:
            coefficients = torch.exp(coefficients_transform)

        return coefficients

    def inverse_log_power(self, coefficients_transform: torch.Tensor) -> torch.Tensor:

        coefficients = None
        if self.representation_name == RepresentationName.lca:
            coefficients = torch.sign(coefficients_transform) * \
                           (torch.pow(torch.exp(torch.abs(coefficients_transform)), 0.5) - 1.)
        elif self.representation_name == RepresentationName.stft:
            coefficients = torch.pow(torch.exp(coefficients_transform), 0.5)

        return coefficients

    def inverse_log10(self, coefficients_transform: torch.Tensor) -> torch.Tensor:

        coefficients = None
        if self.representation_name == RepresentationName.lca:
            coefficients = torch.sign(coefficients_transform) * (torch.pow(10, torch.abs(coefficients_transform)) - 1.)
        elif self.representation_name == RepresentationName.stft:
            coefficients = torch.pow(10, coefficients_transform)

        return coefficients

    def __call__(self, coefficients: torch.Tensor, dataset_type: str = DataDirectories.noisyspeech_dirname,
                 mode: str = 'transform') -> torch.Tensor:

        valid_mode = ['transform', 'inverse_transform']
        assert mode in valid_mode, f'mode parameter "{mode}" should be in {valid_mode}'

        if dataset_type == DataDirectories.noisyspeech_dirname:
            self.transform_info = self.noisyspeech_transform_info
        elif dataset_type == DataDirectories.cleanspeech_dirname:
            self.transform_info = self.cleanspeech_transform_info

        if mode == 'transform':
            for i, transform in enumerate(self.transform_fn):
                coefficients = transform(coefficients)
                if transform.__name__ in ['log_power', 'log10'] and transform.__name__ not in self.transform_metadata:
                    for k, v in self.transform_info.items():
                        self.transform_info[k] = transform(torch.tensor([v],
                                                                        dtype=coefficients.dtype,
                                                                        device=coefficients.device))
                    self.transform_metadata.append(transform.__name__)

        elif mode == 'inverse_transform':
            for i, inverse_transform in enumerate(self.inverse_transform_fn):
                coefficients = inverse_transform(coefficients)
                if inverse_transform.__name__[8:] in ['log_power', 'log10'] and inverse_transform.__name__[8:] in self.transform_metadata:
                    for k, v in self.transform_info.items():
                        self.transform_info[k] = inverse_transform(torch.tensor([v],
                                                                                dtype=coefficients.dtype,
                                                                                device=coefficients.device))
                    self.transform_metadata.remove(inverse_transform.__name__[8:])

        return coefficients


