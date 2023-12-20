"""
File:
    Visualization/VisualizationManager.py

Description:
    Defines the VisualizationManager class.
"""


try:
    from src.lca.FilterBankManager import FilterBankManager
    from src.lca.constants import FilterBankParameters
except ImportError:
    pass


import os
import numpy as np
import torchaudio
from pathlib import Path
from comet_ml import Experiment
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Optional
from src.data.constants import AudioParameters
from src.model.constants import SNNstr, ANNstr
from src.data.constants import RepresentationName


class VisualizationManager:
    """Class that handles data visualization.
    """

    def __init__(self, fontsize: int = 15, figsize_signal: tuple = (16, 10), figsize_snn: tuple = (20, 20),
                 nb_xticklabels: int = 10, nb_yticklabels: int = 8) -> None:

        # self.legend_loc = 'best'
        self.fontsize = fontsize
        self.figsize_signal = figsize_signal
        self.figsize_snn = figsize_snn
        self.nb_xticklabels = nb_xticklabels
        self.nb_yticklabels = nb_yticklabels

    def plot_signal(self, signal: Union[np.ndarray, List[np.ndarray]], signal_name: str, sample_rate: int,
                    signal_label: List[str] = [], alpha: float = 0.5, show_flag: bool = True,
                    fig_dir: Optional[str] = None, fig_format: str = 'jpg') -> None:
        """Method that plots the audio signal.

        Parameters
        ----------
        signal: np.ndarray
            A numpy array of audio signal.
        signal_name: str
            Audio signal name.
        sample_rate: int
            Sampling rate.
        signal_label: List[str]
            Signal label.
        alpha: float
            Sacalar value to adjust the transparency.
        show_flag: bool
            Boolean that indicates weather to show figure.
        fig_dir: Optional[str]
            Path to save figure.
        fig_format: str
            Format for figure saving.
        """

        sns.set_theme()
        plt.figure(figsize=self.figsize_signal)
        if isinstance(signal, np.ndarray):
            librosa.display.waveshow(signal, sr=sample_rate, alpha=alpha)
        else:
            for i in range(len(signal)):
                librosa.display.waveshow(signal[i], sr=sample_rate, alpha=alpha, label=signal_label[i])
            plt.legend(loc='upper right')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(signal_name, fontsize=self.fontsize)
        plt.tight_layout()

        # Save the figure
        if fig_dir:
            plt.savefig(f"{fig_dir}.{fig_format}")

        # Show the plot
        if show_flag:
            plt.show()

        plt.close()

    def plot_spectrogram(self, coefficients: np.ndarray, transform_name: Optional[str] = 'log_power',
                         z_max: Optional[int] = 64, channel_idx: bool = False, collated: bool = False,
                         experiment: Optional[Experiment] = None, epoch: int = None, signal_type: str = None,
                         epsilon: float = 1e-8, cmap: str = 'turbo', vmax: Optional[float] = None,
                         vmin: Optional[float] = None, plot_title: str = 'Spectrogram (dB)', show_flag: bool = True,
                         fig_dir: Optional[str] = None, fig_format: str = 'jpg') -> None:
        """Method that plots the STFT magnitude.

        Parameters
        ----------
        coefficients: np.ndarray
            A numpy array of STFT magnitude coefficients.
        transform_name: Optional[str]
            Transform function name.
        z_max: Optional[int]
            Maximum number of channels if input has 3 dimensions.
        channel_idx: int
            Boolean that indicates weather to show channel index.
        collated: bool
            Boolean that indicates weather to collate multiple figures.
        experiment: Optional[Experiment]
            Comet ML experiment instance.
        epoch: int
            Training iteration.
        signal_type: str
            Signal type.
        epsilon: float
            A small value to avoid computation error.
        cmap: str
            Colormap.
        vmax: Optional[float]
            Maximum value of figure data range.
        vmin: Optional[float]
            Minimum value of figure data range.
        plot_title: str
            Plot title.
        show_flag: bool
            Boolean that indicates weather to show figure.
        fig_dir: Optional[str]
            Path to save figure.
        fig_format: str
            Format for figure saving.
        """

        valid_transform_name = [None, 'log10', 'log_power', 'abs']
        assert transform_name in valid_transform_name, \
            f'transform_name parameter "{transform_name}" should be in {valid_transform_name}'

        if transform_name == 'log10':
            coefficients_magnitude = np.abs(coefficients)
            spectrogram = 20 * np.log10(np.maximum(coefficients_magnitude, epsilon))
            # vmax, vmin = 0, -100
            vmax, vmin = np.min(spectrogram), np.max(spectrogram)
        elif transform_name == 'log_power':
            coefficients_magnitude = np.abs(coefficients)
            spectrogram = np.log(np.maximum(np.power(coefficients_magnitude, 2), epsilon))
            vmax, vmin = np.min(spectrogram), np.max(spectrogram)
        elif transform_name == 'abs':
            spectrogram = np.abs(coefficients)
            vmax, vmin = np.min(spectrogram), np.max(spectrogram)
        elif transform_name is None:
            spectrogram = coefficients
            vmin, vmax = np.min(spectrogram), np.max(spectrogram)

        if signal_type:
            plot_title = signal_type + ' ' + plot_title

        sns.set_theme()
        if spectrogram.ndim == 2:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize_signal)
            img = ax.imshow(spectrogram, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
            # fig.suptitle(plot_title)
            ax.set_title(plot_title, fontsize=self.fontsize)
            ax.set_xlabel('Time frame index')
            ax.set_ylabel('Frequency channel bin index')
            ax.grid(False)
            plt.colorbar(img, ax=ax)
            plt.tight_layout()
        elif spectrogram.ndim == 3:
            z = spectrogram.shape[0] if z_max is None else min(z_max, spectrogram.shape[0])
            # Square grid
            nrows = int(np.ceil(np.sqrt(z)))
            ncols= int(np.ceil(z / nrows))
            # Custom aspect ratio
            # nrows = int(np.ceil(np.sqrt(z * 2))) if z > 1 else 1
            # ncols = int(np.ceil(z / nrows))
            fig, ax = plt.subplots(nrows, ncols, figsize=self.figsize_signal, squeeze=False, sharex='all', sharey='all')
            for i in range(z):
                row_idx = i // ncols
                col_idx = i % ncols
                img = ax[row_idx, col_idx].imshow(spectrogram[i, :, :],
                                                  cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
                if channel_idx:
                    ax[row_idx, col_idx].set_title(f'Channel index: {i}', fontsize=8)
                    plt.subplots_adjust(wspace=0.05, hspace=0.2)
                else:
                    if collated:
                        plt.subplots_adjust(wspace=0., hspace=0.)
                    else:
                        plt.subplots_adjust(wspace=0.01, hspace=0.03)
                # ax[row_idx, col_idx].axis('off')
                ax[row_idx, col_idx].grid(False)

            for k in range(z, nrows*ncols):
                row_idx, col_idx = k // ncols, k % ncols
                # ax[row_idx, col_idx].set_visible(False)
                fig.delaxes(ax[row_idx, col_idx])

            fig.suptitle(plot_title, fontsize=self.fontsize)
            fig.text(0.5, 0.03, 'Time frame index', ha='center', fontsize=self.fontsize)
            fig.text(0.04, 0.5, 'Frequency channel bin index', va='center', rotation='vertical', fontsize=self.fontsize)

            plt.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
            fig.colorbar(img, cax=cbar_ax)

        # Save the figure
        if fig_dir:
            plt.savefig(f"{fig_dir}.{fig_format}")

        # Log the figure
        if experiment:
            if epoch is not None:
                experiment.log_figure('Train: ' + plot_title, figure=plt, step=epoch)
            else:
                experiment.log_figure('Test: ' + plot_title, figure=plt, step=None)

        # Show the plot
        if show_flag:
            plt.show()

        plt.close()

    def plot_coefficients(self, coefficients: np.ndarray, signal_type: str = None, spec_cmap: str = 'turbo',
                          plot_title: str = 'coefficients', show_flag: bool = True, fig_dir: Optional[str] = None,
                          fig_format: str = 'jpg') -> None:
        """Method that plots the activation.

        Parameters
        ----------
        coefficients: np.ndarray
            A numpy array of coefficients.
        signal_type: str
            Signal type.
        spec_cmap: str
            Colormap.
        plot_title: str
            Plot title.
        show_flag: bool
            Boolean that indicates weather to show figure.
        fig_dir: Optional[str]
            Path to save figure.
        fig_format: str
            Format for figure saving.
        """

        sns.set_theme(style='whitegrid')
        fig, ax = plt.subplots(figsize=self.figsize_signal)
        img = plt.imshow(coefficients, origin='lower', aspect='auto', cmap=spec_cmap)  # , vmin=-1, vmax=1)

        if signal_type:
            plot_title = signal_type + ' ' + plot_title

        # Define filterbank
        filterbank = FilterBankManager(num_channels=FilterBankParameters.num_channels.value,
                                       ker_len=FilterBankParameters.ker_len.value)
        filterbank.fs = AudioParameters.sample_rate

        # ax.set_xticks(np.linspace(0, coefficients.shape[1], 10))
        # ax.set_xticklabels(['%.1f' % i for i in np.linspace(0, AudioParameters.max_duration -
        #                                                     1 / AudioParameters.sample_rate, 10)])
        ax.set_yticks(np.linspace(0, len(filterbank.central_freq) - 1, num=self.nb_yticklabels, dtype=int))
        ax.set_yticklabels(['%.0f' % filterbank.central_freq.flip(0)[i] for i in np.linspace(
            0, len(filterbank.central_freq) - 1, num=self.nb_yticklabels, dtype=int)
                            ])

        ax.set_title(plot_title, fontsize=self.fontsize)
        ax.set_xlabel('Discrete time index')
        ax.set_ylabel('Frequency channels (Hz)')
        cbar = ax.figure.colorbar(img, ax=ax, format="%+0.1f")
        cbar.ax.set_ylabel('Density')

        # for i in range(FilterBankParameters.num_channels.value - 1):
        #     ax.axhline(i + 0.5, color='w', linewidth=0.5)

        plt.tight_layout()

        # Save the figure
        if fig_dir:
            plt.savefig(f"{fig_dir}.{fig_format}")

        # Show the plot
        if show_flag:
            plt.show()

        plt.close()

    def plot_dist(self, coefficients: torch.Tensor, experiment: Optional[Experiment] = None, signal_type: str = None,
                  bins: int = 50, kde: bool = False, log_scale: bool = False, plot_title: str = 'distribution',
                  show_flag: bool = True, fig_dir: Optional[str] = None, fig_format: str = 'jpg') -> None:
        """Method that plots data distribution.

        Parameters
        ----------
        coefficients: torch.Tensor
            A numpy array of coefficients.
        experiment: Optional[Experiment]
            Comet ML experiment instance.
        signal_type: str
            Signal type.
        bins: int
            Number of bins.
        kde: bool
            Boolean that indicates weather to compute a kernel density estimate.
        log_scale: bool
            Boolean that indicates weather to set axis scale(s) to log.
        plot_title: str
            Plot title.
        show_flag: bool
            Boolean that indicates weather to show figure.
        fig_dir: Optional[str]
            Path to save figure.
        fig_format: str
            Format for figure saving.
        """
        sns.set_theme()
        plt.figure(figsize=self.figsize_signal)

        if signal_type:
            plot_title = signal_type + ' ' + plot_title

        sns.histplot(coefficients, bins=bins, kde=kde, log_scale=log_scale)
        plt.title(plot_title, fontsize=self.fontsize)
        plt.tight_layout()

        # Save the figure
        if fig_dir:
            plt.savefig(f"{fig_dir}.{fig_format}")

        # Log the figure
        if experiment:
            experiment.log_figure(plot_title, figure=plt, step=None)

        # Show the plot
        if show_flag:
            plt.show()

        plt.close()

    def plot_raster(self, spikes, scatter_plot: bool = False, z_max: Optional[int] = 64, channel_idx: bool = False,
                    collated: bool = False, experiment: Optional[Experiment] = None, epoch: int = None,
                    signal_type: str = None, plot_title: str = 'Raster plot', show_flag: bool = True,
                    fig_dir: Optional[str] = None, fig_format: str = 'jpg') -> None:
        """Method that plots the raster plot.

        Parameters
        ----------
        spikes: np.ndarray
            Spike trains.
        scatter_plot: bool
            Boolean that indicates weather to use `scatter` function.
        z_max: Optional[int]
            Maximum number of channels if input has 3 dimensions.
        channel_idx: bool
            Boolean that indicates weather to show channel index.
        collated: bool
            Boolean that indicates weather to collate multiple figures.
        experiment: Optional[Experiment]
            Comet ML experiment instance.
        epoch: int
            Training iteration.
        signal_type: str
            Signal type.
        plot_title: str
            Plot title.
        show_flag: bool
            Boolean that indicates weather to show figure.
        fig_dir: Optional[str]
            Path to save figure.
        fig_format: str
            Format for figure saving.
        """

        if scatter_plot:
            plot_title = 'spikegram'

        if signal_type:
            plot_title = signal_type + ' ' + plot_title

        if spikes.ndim == 2:
            sns.set_theme(style='whitegrid')
            fig, ax = plt.subplots(1, 1, figsize=self.figsize_signal)
            if scatter_plot:
                idx_l, idx_c = np.nonzero(spikes)
                ax.scatter(idx_c, idx_l, 1)
                # axs.set_yscale('log')
            else:
                spiking_events = []
                for i in range(spikes.shape[0]):
                    spiking_events.append(np.where(spikes[i, :] != 0.)[0])
                ax.eventplot(spiking_events, orientation='horizontal', color='b')
                ax.set_xticks(np.linspace(0, spikes.shape[1] - 1, num=self.nb_xticklabels, dtype=int))
                ax.set_yticks(np.linspace(0, spikes.shape[0] - 1, num=self.nb_yticklabels, dtype=int))
            ax.set_title(plot_title, fontsize=self.fontsize)
            ax.set_xlabel('Discrete time index')
            ax.set_ylabel('Neuron index')
        elif spikes.ndim == 3:
            z = spikes.shape[0] if z_max is None else min(z_max, spikes.shape[0])
            # Square grid
            nrows = int(np.ceil(np.sqrt(z)))
            ncols = int(np.ceil(z / nrows))
            # Custom aspect ratio
            # nrows = int(np.ceil(np.sqrt(z * 2))) if z > 1 else 1
            # ncols = int(np.ceil(z / nrows))
            # --------
            # scale = 4
            # tuple([scale * k for k in self.figsize_signal])
            fig, ax = plt.subplots(nrows, ncols, figsize=self.figsize_signal, squeeze=False, sharex='all', sharey='all')
            for i in range(z):
                row_idx = i // ncols
                col_idx = i % ncols
                scatter_plot = True
                if scatter_plot:
                    idx_l, idx_c = np.nonzero(spikes[i, :, :])
                    ax[row_idx, col_idx].scatter(idx_c, idx_l, 1)
                else:
                    pass
                if channel_idx:
                    ax[row_idx, col_idx].set_title(f'Channel index: {i}', fontsize=8)
                    plt.subplots_adjust(wspace=0.05, hspace=0.2)
                else:
                    if collated:
                        plt.subplots_adjust(wspace=0., hspace=0.)
                    else:
                        plt.subplots_adjust(wspace=0.01, hspace=0.03)
                # ax[row_idx, col_idx].axis('off')
                # ax[row_idx, col_idx].grid(True)
            for k in range(z, nrows*ncols):
                row_idx, col_idx = k // ncols, k % ncols
                # ax[row_idx, col_idx].set_visible(False)
                fig.delaxes(ax[row_idx, col_idx])

            fig.suptitle(plot_title, fontsize=self.fontsize)
            fig.text(0.5, 0.03, 'Discrete time index', ha='center', fontsize=self.fontsize)
            fig.text(0.04, 0.5, 'Neuron index', va='center', rotation='vertical', fontsize=self.fontsize)

        # fig.set_tight_layout(True)

        # Save the figure
        if fig_dir:
            plt.savefig(f"{fig_dir}.{fig_format}")

        # Log the figure
        if experiment:
            if epoch is not None:
                experiment.log_figure('Train: ' + plot_title, figure=plt, step=epoch)
            else:
                experiment.log_figure('Test: ' + plot_title, figure=plt, step=None)

        # Show the plot
        if show_flag:
            plt.show()

        plt.close()

    def plot_membrane_potential(self, membrane_potential_records: np.ndarray, imshow: bool = True,
                                z_max: Optional[int] = 32, channel_idx: bool = False, collated: bool = False,
                                n_channels: int = 5, experiment: Optional[Experiment] = None, epoch: int = None,
                                signal_type: str = None, cmap: str = 'turbo', plot_title: str = 'Membrane potential',
                                show_flag: bool = True, fig_dir: Optional[str] = None, fig_format: str = 'jpg') -> None:
        """Method that plots the membrane potential records.

            Parameters
            ----------
            membrane_potential_records: np.ndarray
                 Membrane potential records.
            imshow: bool
                Boolean that indicates weather to use `imshow` function.
            z_max: Optional[int]
                Maximum number of channels if input has 3 dimensions.
            channel_idx: bool
                Boolean that indicates weather to show channel index.
            collated: bool
                Boolean that indicates weather to collate multiple figures.
            n_channels: int
                Number of channels within plot.
            experiment: Optional[Experiment]
                Comet ML experiment instance.
            epoch: int
                Training iteration.
            signal_type: str
                Signal type.
            cmap: str
                Colormap.
            plot_title: str
                Plot title.
            show_flag: bool
                Boolean that indicates weather to show figure.
            fig_dir: Optional[str]
                Path to save figure.
            fig_format: str
                Format for figure saving.
            """

        vmin, vmax = np.min(membrane_potential_records), np.max(membrane_potential_records)

        if signal_type:
            plot_title = signal_type + ' ' + plot_title

        sns.set_theme()
        if membrane_potential_records.ndim == 2:
            if imshow:
                fig, ax = plt.subplots(1, 1, figsize=self.figsize_signal)
                img = ax.imshow(membrane_potential_records, cmap=cmap, vmin=vmin, vmax=vmax,
                                origin='lower', aspect='auto')
                ax.set_title(plot_title, fontsize=self.fontsize)
                ax.set_xlabel('Discrete time index')
                ax.set_ylabel('Amplitude')
                plt.colorbar(img, ax=ax)
                plt.tight_layout()
            else:
                fig, ax = plt.subplots(n_channels, figsize=self.figsize_signal, sharex=True, sharey=True)
                for i in range(n_channels):
                    ax[i].plot(membrane_potential_records[i])
                    ax[i].set_title(f'Neuron index: {i}', fontsize=8)
                fig.suptitle(plot_title, fontsize=self.fontsize)
                fig.text(0.5, 0.03, 'Discrete time index', ha='center', fontsize=self.fontsize)
                fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical', fontsize=self.fontsize)
                plt.grid()
        elif membrane_potential_records.ndim == 3:
            z = membrane_potential_records.shape[0] if z_max is None else min(z_max, membrane_potential_records.shape[0])
            # Square grid
            nrows = int(np.ceil(np.sqrt(z)))
            ncols = int(np.ceil(z / nrows))
            # Custom aspect ratio
            # nrows = int(np.ceil(np.sqrt(z * 2))) if z > 1 else 1
            # ncols = int(np.ceil(z / nrows))
            fig, ax = plt.subplots(nrows, ncols, figsize=self.figsize_signal, squeeze=False, sharex='all', sharey='all')
            for i in range(z):
                row_idx = i // ncols
                col_idx = i % ncols
                img = ax[row_idx, col_idx].imshow(membrane_potential_records[i, :, :],
                                                  cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
                if channel_idx:
                    ax[row_idx, col_idx].set_title(f'Channel index: {i}', fontsize=8)
                    plt.subplots_adjust(wspace=0.05, hspace=0.2)
                else:
                    if collated:
                        plt.subplots_adjust(wspace=0., hspace=0.)
                    else:
                        plt.subplots_adjust(wspace=0.01, hspace=0.03)
                # ax[row_idx, col_idx].axis('off')
                # plt.grid(False)
                ax[row_idx, col_idx].grid(False)

            for k in range(z, nrows*ncols):
                row_idx, col_idx = k // ncols, k % ncols
                fig.delaxes(ax[row_idx, col_idx])

            fig.suptitle(plot_title, fontsize=self.fontsize)
            fig.text(0.5, 0.03, 'Discrete time index', ha='center', fontsize=self.fontsize)
            fig.text(0.04, 0.5, 'Neuron index', va='center', rotation='vertical', fontsize=self.fontsize)

            plt.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
            fig.colorbar(img, cax=cbar_ax)

        # Save the figure
        if fig_dir:
            plt.savefig(f"{fig_dir}.{fig_format}")

        # Log the figure
        if experiment:
            if epoch is not None:
                experiment.log_figure('Train: ' + plot_title, figure=plt, step=epoch)
            else:
                experiment.log_figure('Test: ' + plot_title, figure=plt, step=None)

        # Show the plot
        if show_flag:
            plt.show()

        plt.close()

    def plot_signal_data(self, audio_dir: Union[str, Path], index: int, plots_dir: Union[str, Path],
                         representation_name: str, coefficients: np.ndarray, signal_type: str,
                         experiment: Optional[Experiment] = None, scatter_plot: bool = False,
                         plot_coefficients: bool = True, show_flag: bool = False, example_fig_format: str = 'jpg',
                         verbose: bool = False) -> None:
        """Method that plots signal data: audio waveform and data representation.

        Parameters
        ----------
        audio_dir: Union[str, Path]
            Directory of the audio dataset.
        index: int
            Signal index.
        plots_dir: Union[str, Path]
            Plots directory.
        representation_name: str
            Representation name.
        coefficients: np.ndarray
            Representation coefficients.
        signal_type: str
            Signal type.
        experiment: Optional[Experiment]
            Comet ML experiment instance.
        scatter_plot: bool
            Boolean that indicates weather to use `scatter` function.
        plot_coefficients: bool
            Boolean that indicates weather to plot representation coefficients.
        show_flag: bool
            Boolean that indicates weather to show figure.
        example_fig_format: str
            Format for figure saving.
        verbose: bool
            Boolean that indicates weather to print specific output.
        """

        # --------------------------------------------------- Audio ----------------------------------------------------
        # Load signal
        list_audio = sorted(os.listdir(audio_dir))
        # audio_index = int(index*dist.get_world_size() + dist.get_rank())
        audio_index = index
        file_name = list_audio[audio_index]
        audio, sr = torchaudio.load(os.path.join(audio_dir, file_name))

        if sr != AudioParameters.sample_rate:
            audio = torchaudio.transforms.Resample(sr, AudioParameters.sample_rate)(audio)

        audio = librosa.util.normalize(audio.view(audio.shape[-1]).numpy(), axis=0)

        # Plot signal waveform
        audio_name = signal_type + ' audio'
        audio_fig_dir = os.path.join(plots_dir, audio_name)
        signal_name = str(file_name[:-(len(AudioParameters.audio_type) + 1)])
        self.plot_signal(signal=audio,
                         signal_name=signal_name,
                         sample_rate=AudioParameters.sample_rate,
                         show_flag=show_flag,
                         fig_dir=audio_fig_dir,
                         fig_format=example_fig_format)

        # Log signal waveform
        if experiment:
            experiment.log_image(audio_fig_dir + '.' + example_fig_format,
                                 name=audio_name, image_format=example_fig_format)
        # Log signal audio
        aud_dir = os.path.join(audio_dir, file_name)
        if experiment:
            experiment.log_audio(aud_dir, file_name=f'{signal_type.split()[0]} {file_name}')

        # ---------------------------------------------- LCA coefficients ----------------------------------------------
        if representation_name == RepresentationName.lca:
            if plot_coefficients:
                # Plot coefficients
                coefficients_name = signal_type + ' spikegram'
                coefficients_fig_dir = os.path.join(plots_dir, coefficients_name)
                self.plot_raster(spikes=coefficients,
                                 scatter_plot=scatter_plot,
                                 signal_type=signal_type,
                                 show_flag=show_flag,
                                 fig_dir=coefficients_fig_dir,
                                 fig_format=example_fig_format)
                # Log coefficients
                if experiment:
                    experiment.log_image(coefficients_fig_dir + '.' + example_fig_format,
                                         name=coefficients_name, image_format=example_fig_format)

                if verbose:
                    print('\t - Nb {} coefficients = {}'.format(signal_type, np.count_nonzero(coefficients)))

        # --------------------------------------------- STFT coefficients ----------------------------------------------
        elif representation_name == RepresentationName.stft:
            if plot_coefficients:
                # Plot coefficients
                coefficients_name = signal_type + ' STFT coefficients'
                coefficients_fig_dir = os.path.join(plots_dir, coefficients_name)
                self.plot_spectrogram(coefficients=coefficients,
                                      transform_name='log10',
                                      signal_type=signal_type,
                                      show_flag=show_flag,
                                      fig_dir=coefficients_fig_dir,
                                      fig_format=example_fig_format)
                # Log coefficients
                if experiment:
                    experiment.log_image(coefficients_fig_dir + '.' + example_fig_format,
                                         name=coefficients_name, image_format=example_fig_format)

                if verbose:
                    print('\t - Nb {} coefficients = {}'.format(signal_type, np.count_nonzero(coefficients)))

    def plot_snn_layers(self, spk_rec: List[np.array], mem_rec: List[np.array], mem: torch.Tensor, x_data: torch.Tensor,
                        y_data: torch.Tensor, output: List[torch.Tensor], plot_hidden: bool = True,
                        z_max: Optional[int] = 16, scatter_plot: bool = False, imshow: bool = True, data_idx: int = 0,
                        experiment: Optional[Experiment] = None, epoch: int = None, show_flag: bool = False) -> None:
        """Method that plots output of SNN layers.

        Parameters
        ----------
        spk_rec: List[np.array]
            List of spike trains of SNN layers.
        mem_rec: List[np.array]
            List of membrane potential of SNN layers.
        mem: torch.Tensor
            SNN output.
        x_data: torch.Tensor
            SNN input.
        y_data: torch.Tensor
            SNN target output.
        output: List[torch.Tensor]
            SNN intermediate output if `use_intermediate_output` parameter is True.
        plot_hidden: bool
            Boolean that indicates weather to plot output of hidden layers.
        z_max: Optional[int]
            Maximum number of channels if input has 3 dimensions.
        scatter_plot: bool
            Boolean that indicates weather to use `scatter` function.
        imshow: bool
            Boolean that indicates weather to use `imshow` function.
        data_idx: int
            Data index.
        experiment: Optional[Experiment]
            Comet ML experiment instance.
        epoch: int
            Training iteration.
        show_flag: bool
            Boolean that indicates weather to show figure.
        """

        if plot_hidden:
            for i in range(len(spk_rec)):

                signal_type = 'Input layer' if i == 0 else 'Spiking layer' + ' ' + str(format(i, "02d"))
                spk_rec_i = spk_rec[i][data_idx]
                if i == 0:
                    self.plot_membrane_potential(membrane_potential_records=spk_rec_i,
                                                 imshow=imshow,
                                                 z_max=z_max,
                                                 experiment=experiment,
                                                 epoch=epoch,
                                                 signal_type=signal_type,
                                                 show_flag=show_flag,
                                                 plot_title='Input current'
                                                 )
                self.plot_raster(spikes=spk_rec_i,
                                 scatter_plot=scatter_plot,
                                 z_max=z_max,
                                 experiment=experiment,
                                 epoch=epoch,
                                 signal_type=signal_type,
                                 show_flag=show_flag)

                signal_type = f'Spiking layer {format(i + 1, "02d")}' if i < len(spk_rec) - 1 else 'Readout layer output'
                mem_rec_i = mem_rec[i][data_idx]
                self.plot_membrane_potential(membrane_potential_records=mem_rec_i,
                                             imshow=imshow,
                                             z_max=z_max,
                                             experiment=experiment,
                                             epoch=epoch,
                                             signal_type=signal_type,
                                             show_flag=show_flag
                                             )

        plot_title = ''

        # Output
        signal_type = '__all__ output'
        mem_i = mem.detach().clone().cpu().numpy()[data_idx]
        self.plot_membrane_potential(membrane_potential_records=mem_i,
                                     imshow=imshow,
                                     experiment=experiment,
                                     epoch=epoch,
                                     signal_type=signal_type,
                                     show_flag=show_flag
                                     )

        # Input
        signal_type = '__all__ Input'
        x_data_i = x_data.detach().clone().cpu().numpy()[data_idx]
        self.plot_membrane_potential(membrane_potential_records=x_data_i,
                                     imshow=imshow,
                                     experiment=experiment,
                                     epoch=epoch,
                                     signal_type=signal_type,
                                     show_flag=show_flag
                                     )

        # True output
        signal_type = '__all__ True output'
        y_data_i = y_data.detach().clone().cpu().numpy()[data_idx]
        self.plot_membrane_potential(membrane_potential_records=y_data_i,
                                     imshow=imshow,
                                     experiment=experiment,
                                     epoch=epoch,
                                     signal_type=signal_type,
                                     show_flag=show_flag
                                     )

        # Rescaled intermediate layers output
        for i in range(len(output)):
            signal_type = f'Intermediate output (idx={format(i, "02d")})'
            output_i = output[i].detach().clone().cpu().numpy()[data_idx]
            self.plot_membrane_potential(membrane_potential_records=output_i,
                                         imshow=imshow,
                                         experiment=experiment,
                                         epoch=epoch,
                                         signal_type=signal_type,
                                         show_flag=show_flag
                                         )

    def plot_ann_layers(self, out_rec: List[np.array], out: torch.Tensor, x_data: torch.Tensor, y_data: torch.Tensor,
                        model_name: str, plot_hidden: bool = True, z_max: Optional[int] = 64,
                        plot_title: str = 'output', imshow: bool = True, data_idx: int = 0,
                        experiment: Optional[Experiment] = None, epoch: int = None, show_flag: bool = False) -> None:
        """Method that plots output of SNN layers.

        Parameters
        ----------
        out_rec: List[np.array]
            List of output of ANN layers.
        out: torch.Tensor
            ANN output.
        x_data: torch.Tensor
            ANN input.
        y_data: torch.Tensor
            ANN target output.
        model_name: str
            ANN model name.
        plot_hidden: bool
            Boolean that indicates weather to plot output of hidden layers.
        z_max: Optional[int]
            Maximum number of channels if input has 3 dimensions.
        plot_title: str
            Plot title.
        imshow: bool
            Boolean that indicates weather to use `imshow` function.
        data_idx: int
            Data index.
        experiment: Optional[Experiment]
            Comet ML experiment instance.
        epoch: int
            Training iteration.
        show_flag: bool
            Boolean that indicates weather to show figure.
        """

        # Plot ANN layers output
        if plot_hidden:
            for i in range(len(out_rec)):

                if i == 0:
                    signal_type = 'Input layer'
                elif i == (len(out_rec) - 1):
                    signal_type = 'Output layer'
                else:
                    signal_type = 'Hidden layer' + ' ' + str(format(i, "02d"))

                out_rec_i = out_rec[i][data_idx]

                if len(out_rec) > i and len(out_rec_i.shape) == 3 and model_name in [ANNstr.CNNstr,
                                                                                     ANNstr.UNetstr,
                                                                                     ANNstr.ResBottleneckUNetstr]:

                    # CSNN
                    self.plot_membrane_potential(membrane_potential_records=out_rec_i,
                                                 imshow=imshow,
                                                 z_max=z_max,
                                                 experiment=experiment,
                                                 epoch=epoch,
                                                 signal_type=signal_type,
                                                 plot_title=plot_title,
                                                 show_flag=show_flag
                                                 )
                else:
                    # FCSNN
                    self.plot_membrane_potential(membrane_potential_records=out_rec_i,
                                                 imshow=imshow,
                                                 z_max=z_max,
                                                 experiment=experiment,
                                                 epoch=epoch,
                                                 signal_type=signal_type,
                                                 plot_title=plot_title,
                                                 show_flag=show_flag)

        plot_title = ''

        # Output
        signal_type = '__all__ output'
        out_i = out.detach().clone().cpu().numpy()[data_idx]
        self.plot_membrane_potential(membrane_potential_records=out_i,
                                     imshow=imshow,
                                     experiment=experiment,
                                     epoch=epoch,
                                     signal_type=signal_type,
                                     plot_title=plot_title,
                                     show_flag=show_flag)

        # Input
        signal_type = '__all__ input'
        x_data_i = x_data.detach().clone().cpu().numpy()[data_idx]
        self.plot_membrane_potential(membrane_potential_records=x_data_i,
                                     imshow=imshow,
                                     experiment=experiment,
                                     epoch=epoch,
                                     signal_type=signal_type,
                                     plot_title=plot_title,
                                     show_flag=show_flag)

        # True output
        signal_type = '__all__ True output'
        y_data_i = y_data.detach().clone().cpu().numpy()[data_idx]
        self.plot_membrane_potential(membrane_potential_records=y_data_i,
                                     imshow=imshow,
                                     experiment=experiment,
                                     epoch=epoch,
                                     signal_type=signal_type,
                                     plot_title=plot_title,
                                     show_flag=show_flag)

    def plot_loss(self, training_loss_hist: List[float], validation_loss_hist: List[float] = None,
                  plot_title: str = 'Loss per epoch', show_flag: bool = True, fig_dir: Optional[str] = None,
                  fig_format: str = 'jpg') -> None:
        """Method that plots the loss history.

        Parameters
        ----------
        training_loss_hist: List[float]
            Training loss history list.
        validation_loss_hist: List[float]
            Validation loss history list.
        plot_title: str
            Plot title.
        show_flag: bool
            Boolean that indicates weather to show figure.
        fig_dir: Optional[str]
            Path to save figure.
        fig_format: str
            Format for figure saving.
        """

        sns.set_theme(style='whitegrid')
        plt.figure(figsize=self.figsize_signal)
        plt.plot(training_loss_hist, label='Training loss')
        if validation_loss_hist:
            plt.plot(validation_loss_hist, label='Validation loss')
        plt.xticks(np.linspace(0, len(training_loss_hist), 11, dtype=int))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(plot_title, fontsize=self.fontsize)
        plt.legend(loc='upper right')
        plt.tight_layout()

        # Save the figure
        if fig_dir:
            plt.savefig(f"{fig_dir}.{fig_format}")

        # Show the plot
        if show_flag:
            plt.show()

        plt.close()

    def plot_perceptual_metric(self, training_perceptual_metric_hist: List[float],
                               validation_perceptual_metric_hist: List[float] = None,
                               show_flag: bool = True, fig_dir: Optional[str] = None, fig_format: str = 'jpg') -> None:
        """Method that plots a perceptual metric history.

        Parameters
        ----------
        training_perceptual_metric_hist: List[float]
            Training perceptual metric history list.
        validation_perceptual_metric_hist: List[float]
            Validation perceptual metric history list.
        show_flag: bool
            Boolean that indicates weather to show figure.
        fig_dir: Optional[str]
            Path to save figure.
        fig_format: str
            Format for figure saving.
        """

        sns.set_theme(style='whitegrid')
        plt.figure(figsize=self.figsize_signal)
        plt.plot(training_perceptual_metric_hist, label='Training perceptual metric')
        plt.plot(validation_perceptual_metric_hist, label='Validation perceptual metric')
        plt.xticks(np.linspace(0, len(training_perceptual_metric_hist), 11, dtype=int))
        plt.xlabel('Epoch')
        plt.ylabel('Perceptual metric')
        plt.title('Perceptual_metric per epoch', fontsize=self.fontsize)
        plt.legend(loc='upper right')
        plt.tight_layout()

        # Save the figure
        if fig_dir:
            plt.savefig(f"{fig_dir}.{fig_format}")

        # Show the plot
        if show_flag:
            plt.show()

        plt.close()

    def _plot_kernel(self, weights: np.array, sample_rate: int, show_flag: bool = True, fig_dir: Optional[str] = None,
                    fig_format: str = 'jpg') -> None:
        """Method that plots FilterBank kernel.

        Parameters
        ----------
        weights: np.array
            GT FilterBank weights.
        sample_rate: int
            Sampling rate.
        show_flag: bool
            Boolean that indicates weather to show figure.
        fig_dir: Optional[str]
            Path to save figure.
        fig_format: str
            Format for figure saving.
        """

        num_channels = weights.shape[0]

        sns.set_theme(style='whitegrid')
        plt.figure(figsize=self.figsize_signal)

        for i in range(num_channels):
            plt.magnitude_spectrum(weights[i], Fs=fs, scale='dB')

        plt.xscale('log')
        plt.xlim(right=sample_rate / 2)
        plt.ylim(top=-20, bottom=-100)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.title('cGC Filter Bank magnitude in dB', fontsize=self.fontsize)
        plt.tight_layout()

        # Save the figure
        if fig_dir:
            plt.savefig(f"{fig_dir}.{fig_format}")

        # Show the plot
        if show_flag:
            plt.show()

        plt.close()

