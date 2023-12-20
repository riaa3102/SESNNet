"""
File:
    model/LossManager.py

Description:
    Defines the Loss function.
"""


from typing import Optional
import librosa
import numpy as np
from src.model.utils import pad_coefficients, complex_spectrogram
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pesq import pesq
from pystoi import stoi
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
import speechbrain
from speechbrain.nnet.loss.stoi_loss import stoi_loss as stoi_loss_fn
# from torch_stoi import NegSTOILoss as stoi_loss_fn
from speechbrain.nnet.loss.si_snr_loss import si_snr_loss as si_snr_loss_fn
from src.data.TransformManager import TransformManager
from src.lca.constants import LcaParameters
from src.stft.constants import StftParameters
from src.data.constants import AudioParameters, RepresentationName


class LossManager(nn.Module):
    """Class that implements the loss function.
    """

    def __init__(self,
                 representation_name: str,
                 loss_name: list[str],
                 loss_weight: list[float],
                 loss_bias: list[float],
                 metric_name: str,
                 weights: torch.Tensor,
                 transform_manager: TransformManager,
                 reduction: str = 'mean',
                 nb_digits: int = 5
                 ) -> None:

        super(LossManager, self).__init__()

        self.representation_name = representation_name

        self.loss_name = loss_name
        self.loss_weight = loss_weight
        self.loss_bias = loss_bias

        self.weights = weights
        self.transform_manager = transform_manager
        self.metric_name = metric_name
        self.reduction = reduction
        self.nb_digits = nb_digits

    def forward(self, input_: torch.Tensor, target: torch.Tensor, input_phase: Optional[torch.Tensor] = None,
                target_phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Method that computes loss function.

        Parameters
        ----------
        input_: torch.Tensor
            Model output tensor.
        target: torch.Tensor
            target tensor.
        input_phase: torch.Tensor
            input phase if encoder is STFT.
        target_phase: torch.Tensor
            input phase if encoder is STFT.
        """

        loss = self.loss_weight[0] * (
                        getattr(self, self.loss_name[0])(input_, target, input_phase, target_phase) + self.loss_bias[0])

        if len(self.loss_name) > 1:
            for i in range(1, len(self.loss_name)):
                loss = loss + self.loss_weight[i] * (
                        getattr(self, self.loss_name[i])(input_, target, input_phase, target_phase) + self.loss_bias[i])
        return loss

    def mse_loss(self, input_: torch.Tensor, target: torch.Tensor, input_phase: Optional[torch.Tensor] = None,
                 target_phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Method that computes MSE loss function.

        Parameters
        ----------
        input_: torch.Tensor
            Model output tensor.
        target: torch.Tensor
            target tensor.
        input_phase: torch.Tensor
            input phase if encoder is STFT.
        target_phase: torch.Tensor
            input phase if encoder is STFT.
        """
        mse_loss_fn = nn.MSELoss(reduction=self.reduction)
        return mse_loss_fn(input_, target)

    def l1_loss(self, input_: torch.Tensor, target: torch.Tensor, input_phase: Optional[torch.Tensor] = None,
                 target_phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Method that computes l1 loss function.

        Parameters
        ----------
        input_: torch.Tensor
            Model output tensor.
        target: torch.Tensor
            target tensor.
        input_phase: torch.Tensor
            input phase if encoder is STFT.
        target_phase: torch.Tensor
            input phase if encoder is STFT.
        """
        l1_loss_fn = nn.L1Loss(reduction=self.reduction)
        return l1_loss_fn(input_, target)

    def lsd_loss(self, input_: torch.Tensor, target: torch.Tensor, input_phase: Optional[torch.Tensor] = None,
                 target_phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Method that computes LSD loss function.

        Parameters
        ----------
        input_: torch.Tensor
            Model output tensor.
        target: torch.Tensor
            target tensor.
        input_phase: torch.Tensor
            input phase if encoder is STFT.
        target_phase: torch.Tensor
            input phase if encoder is STFT.
        """
        lsd = torch.mean(torch.sqrt(torch.mean((target - input_)**2, dim=-2)), dim=-1)
        return torch.mean(lsd)

    def time_mse_loss(self, input_: torch.Tensor, target: torch.Tensor, input_phase: Optional[torch.Tensor] = None,
                 target_phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Method that computes time-domain MSE loss function.

        Parameters
        ----------
        input_: torch.Tensor
            Model output tensor.
        target: torch.Tensor
            target tensor.
        input_phase: torch.Tensor
            input phase if encoder is STFT.
        target_phase: torch.Tensor
            input phase if encoder is STFT.
        """
        mse_loss_fn = nn.MSELoss(reduction=self.reduction)
        return mse_loss_fn(self._rec_fn(input_, input_phase), self._rec_fn(target, target_phase))

    def huber_loss(self, input_: torch.Tensor, target: torch.Tensor, input_phase: Optional[torch.Tensor] = None,
                 target_phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Method that computes Huber loss function.

        Parameters
        ----------
        input_: torch.Tensor
            Model output tensor.
        target: torch.Tensor
            target tensor.
        input_phase: torch.Tensor
            input phase if encoder is STFT.
        target_phase: torch.Tensor
            input phase if encoder is STFT.
        """
        huber_loss_fn = nn.HuberLoss(reduction=self.reduction)
        return huber_loss_fn(input_, target)

    def stoi_loss(self, input_: torch.Tensor, target: torch.Tensor, input_phase: Optional[torch.Tensor] = None,
                 target_phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Method that computes STOI loss function.

        Parameters
        ----------
        input_: torch.Tensor
            Model output tensor.
        target: torch.Tensor
            target tensor.
        input_phase: torch.Tensor
            input phase if encoder is STFT.
        target_phase: torch.Tensor
            input phase if encoder is STFT.
        """

        # stoi_loss_ = stoi_loss_fn(sample_rate=AudioParameters.sample_rate)(self._rec_fn(input_, input_phase),
        #                                                                    self._rec_fn(target, target_phase))

        stoi_loss_ = stoi_loss_fn(self._rec_fn(input_, input_phase),
                                  self._rec_fn(target, target_phase),
                                  torch.ones(input_.shape[0]),
                                  reduction=self.reduction)             # reduction='batch',

        stoi_loss_ = torch.nan_to_num(stoi_loss_)
        return stoi_loss_

    def si_snr_loss(self, input_: torch.Tensor, target: torch.Tensor, input_phase: Optional[torch.Tensor] = None,
                 target_phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Method that computes SI-SNR loss function.

        Parameters
        ----------
        input_: torch.Tensor
            Model output tensor.
        target: torch.Tensor
            target tensor.
        input_phase: torch.Tensor
            input phase if encoder is STFT.
        target_phase: torch.Tensor
            input phase if encoder is STFT.
        """

        si_snr_loss_ = si_snr_loss_fn(self._rec_fn(input_, input_phase),
                                      self._rec_fn(target, target_phase),
                                      torch.ones(input_.shape[0]),
                                      reduction=self.reduction)
        return si_snr_loss_

    def si_sdr_loss(self, input_: torch.Tensor, target: torch.Tensor, input_phase: Optional[torch.Tensor] = None,
                 target_phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Method that computes SI-SDR loss function.

        Parameters
        ----------
        input_: torch.Tensor
            Model output tensor.
        target: torch.Tensor
            target tensor.
        input_phase: torch.Tensor
            input phase if encoder is STFT.
        target_phase: torch.Tensor
            input phase if encoder is STFT.
        """

        rec_input_ = self._rec_fn(input_, input_phase)
        rec_target = self._rec_fn(target, target_phase)

        si_sdr = 0.
        for i in range(input_.shape[0]):
            si_sdr += scale_invariant_signal_distortion_ratio(rec_input_[i], rec_target[i])

        if self.reduction == 'mean':
            si_sdr = si_sdr / input_.shape[0]

        return si_sdr

    def perceptual_metric(self, input_: torch.Tensor, target: torch.Tensor, input_phase: Optional[torch.Tensor] = None,
                 target_phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Method that computes a perceptual metric as accuracy.

        Parameters
        ----------
        input_: torch.Tensor
            Model output tensor.
        target: torch.Tensor
            target tensor.
        input_phase: torch.Tensor
            input phase if encoder is STFT.
        target_phase: torch.Tensor
            input phase if encoder is STFT.
        """

        metric = 0.
        if self.metric_name == 'pesq':
            metric = self.pesq_fn(self._rec_fn(input_, input_phase), self._rec_fn(target, target_phase))
        elif self.metric_name == 'stoi':
            metric = self.stoi_fn(self._rec_fn(input_, input_phase), self._rec_fn(target, target_phase))
        else:
            assert self.metric_name in ['pesq', 'stoi'], 'metric_name parameter must be "pesq" or "stoi"'

        return metric

    def pesq_fn(self, rec_input_, rec_target, fs: int = AudioParameters.sample_rate, mode: str = 'wb'):
        """Method that computes PESQ score function.

        Parameters
        ----------
        input_: torch.Tensor
            Model output tensor.
        target: torch.Tensor
            target tensor.
        input_phase: torch.Tensor
            input phase if encoder is STFT.
        target_phase: torch.Tensor
            input phase if encoder is STFT.
        """

        pesq_list = []
        for i in range(rec_input_.shape[0]):
            rec_input_i = rec_input_.clone().detach().cpu()[i]
            rec_target_i = rec_target.clone().detach().cpu()[i]

            # Adjust audio length
            min_len = min(rec_input_i.shape[-1], rec_target_i.shape[-1])
            rec_input_i = rec_input_i.view(rec_input_i.shape[-1])[:min_len]
            rec_target_i = rec_target_i.view(rec_target_i.shape[-1])[:min_len]

            pesq_list.append(pesq(fs, rec_target_i.numpy(), rec_input_i.numpy(), mode))

        return np.round(np.mean(pesq_list), self.nb_digits)

    def stoi_fn(self, rec_input_, rec_target, fs: int = AudioParameters.sample_rate, extended: bool = False):
        """Method that computes STOI score function.

        Parameters
        ----------
        input_: torch.Tensor
            Model output tensor.
        target: torch.Tensor
            target tensor.
        input_phase: torch.Tensor
            input phase if encoder is STFT.
        target_phase: torch.Tensor
            input phase if encoder is STFT.
        """

        stoi_list = []
        for i in range(rec_input_.shape[0]):
            rec_input_i = rec_input_.clone().detach().cpu()[i]
            rec_target_i = rec_target.clone().detach().cpu()[i]

            # Adjust audio length
            min_len = min(rec_input_i.shape[-1], rec_target_i.shape[-1])
            rec_input_i = rec_input_i.view(rec_input_i.shape[-1])[:min_len]
            rec_target_i = rec_target_i.view(rec_target_i.shape[-1])[:min_len]

            stoi_list.append(stoi(rec_target_i.numpy(), rec_input_i.numpy(), fs, extended))

        return np.round(np.mean(stoi_list), self.nb_digits)

    def _rec_fn(self, coefficients: torch.Tensor, coefficients_phase: Optional[torch.Tensor] = None,
               normalize: bool = False) -> torch.Tensor:
        """Method that reconstructs audio signal.

        Parameters
        ----------
        coefficients: torch.Tensor
            Data tensor.
        coefficients_phase: torch.Tensor
            coefficients_phase phase if encoder is STFT.
        normalize: bool
            Boolean that indicates weather to normalize audio data.
        """

        if self.transform_manager:
            coefficients = self.transform_manager(coefficients, mode='inverse_transform')

        if self.representation_name == RepresentationName.lca:
            speech = F.conv_transpose1d(coefficients.view((-1, coefficients.shape[-2], coefficients.shape[-1])),
                                        weight=self.weights,
                                        stride=LcaParameters.stride).squeeze(dim=1)

        elif self.representation_name == RepresentationName.stft:
            if coefficients_phase is None:
                assert coefficients_phase is None, 'coefficients_phase argument is None'

            coefficients_complex = pad_coefficients(complex_spectrogram(coefficients, coefficients_phase))

            speech = torchaudio.transforms.InverseSpectrogram(n_fft=StftParameters.n_fft,
                                                              win_length=StftParameters.win_length,
                                                              hop_length=StftParameters.hop_length,
                                                              normalized=StftParameters.normalized,
                                                              center=StftParameters.center,
                                                              )(coefficients_complex.cpu())

        if normalize and torch.all(torch.isinf(speech) == False):
            speech_list = []
            for i in range(speech.shape[0]):
                speech_i = speech.clone().cpu()[i]
                speech_list.append(torch.tensor(librosa.util.normalize(speech_i.view(-1).numpy(), axis=-1)))
            speech = torch.stack(speech_list, 0)

        return speech.view(speech.shape[0], -1)

