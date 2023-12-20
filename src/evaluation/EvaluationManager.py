"""
File:
    evaluation/EvaluationManager.py

Description:
    Defines the EvaluationManager class.
"""

import concurrent.futures
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from comet_ml import Experiment
from torchmetrics import ScaleInvariantSignalDistortionRatio
from src.data.DatasetManager import DatasetManager
from src.data.constants import AudioParameters, DataDirectories
from src.evaluation.composite import *
from pesq import pesq
from pystoi import stoi
from src.evaluation.DNSMOS.dnsmos_local import DNSMOS


class EvaluationManager:
    """Class that handles enhanced speech evaluation.
    """

    def __init__(self,
                 dataset_manager: DatasetManager,
                 experiment_files_dir: str,
                 experiment: Experiment,
                 use_representation_dir: bool = False,
                 nb_digits: int = 5
                 ) -> None:

        self.dataset_manager = dataset_manager
        self.experiment_files_dir = experiment_files_dir
        self.experiment = experiment
        self.csv_files_dir = os.path.join(self.experiment_files_dir, 'evaluation_csv')
        os.makedirs(self.csv_files_dir)

        if use_representation_dir:
            self.cleanspeech_dir = os.path.join(dataset_manager.representation_reconstruct_dir,
                                                f'{dataset_manager.representation_name.upper()}_audio_reconstruction',
                                                f'{self.dataset_manager.cleanspeech_dirname}_{self.dataset_manager.data_load}')
            self.noisyspeech_dir = os.path.join(dataset_manager.representation_reconstruct_dir,
                                                f'{dataset_manager.representation_name.upper()}_audio_reconstruction',
                                                f'{self.dataset_manager.noisyspeech_dirname}_{self.dataset_manager.data_load}')
        else:
            self.cleanspeech_dir = dataset_manager.cleanspeech_dir
            self.noisyspeech_dir = dataset_manager.noisyspeech_dir

        self.enhancedspeech_dir = dataset_manager.enhancedspeech_dir

        self.dnsmos_model_dir = os.path.join(Path(__file__).parent, 'DNSMOS')

        self.obj_evaluation_dc = {}
        self.composite_obj_evaluation_dc = {}
        self.dnsmos_obj_evaluation_dc = {}
        self.nb_digits = nb_digits

    def compute_evaluation_metrics(self) -> None:
        """Method computes evaluation metrics.
        """

        # Compute PESQ, STOI, ESOI, SI-SNR
        self.compute_objective_eval_metrics(preds_dir=self.enhancedspeech_dir, target_dir=self.cleanspeech_dir,
                                            preds_data_type='enhancedspeech',
                                            filename_prefix=f'{DataDirectories.enhancedspeech_filename}_')

        self.compute_objective_eval_metrics(preds_dir=self.noisyspeech_dir, target_dir=self.cleanspeech_dir,
                                            preds_data_type='noisyspeech')

        # Log objective evaluation scores
        if self.experiment:
            self.experiment.log_metrics(self.obj_evaluation_dc)


        # Compute Csig, Cbak, Covl, segSNR
        self.compute_composite_objective_eval_metrics(preds_dir=self.enhancedspeech_dir,
                                                      target_dir=self.cleanspeech_dir,
                                                      preds_data_type='enhancedspeech',
                                                      filename_prefix=f'{DataDirectories.enhancedspeech_filename}_')

        self.compute_composite_objective_eval_metrics(preds_dir=self.noisyspeech_dir, target_dir=self.cleanspeech_dir,
                                                      preds_data_type='noisyspeech')

        # Log objective evaluation scores
        if self.experiment:
            self.experiment.log_metrics(self.composite_obj_evaluation_dc)


        # Compute SIG, BAK, OVRL, P808_MOS
        audio_files_dir_dict = {'enhancedspeech': self.enhancedspeech_dir,
                                'noisyspeech': self.noisyspeech_dir,
                                'cleanspeech': self.cleanspeech_dir,
                                }

        for data_type, audio_files_dir in audio_files_dir_dict.items():
            self.compute_dnsmos_objective_eval_metrics(speech_dir=audio_files_dir, data_type=data_type)

        # Log Composite objective evaluation scores
        if self.experiment:
            self.experiment.log_metrics(self.dnsmos_obj_evaluation_dc)

    def compute_objective_eval_metrics(self, preds_dir: str, target_dir: str, preds_data_type: str,
                                       filename_prefix: str = '') -> None:
        """Method computes metrics: PESQ, STOI, ESTOI, SI-SNR.

        Parameters
        ----------
        preds_dir: str
            Noisy/Enhanced audio data directory.
        target_dir: str
            Clean audio data directory.
        preds_data_type: str
            'noisy' or 'enhanced' audio.
        filename_prefix: bool
            Enhanced audio data filename prefix.
        """

        list_speech_audio = sorted(os.listdir(preds_dir))
        if len(filename_prefix) > 0:
            list_speech_audio = [list_speech_audio[i][len(filename_prefix):] for i in range(len(list_speech_audio))]

        pesq_list = []
        stoi_list = []
        estoi_list = []
        # si_sdr_list = []
        si_snr_list = []
        for file in tqdm(list_speech_audio, leave=True, desc='Objective Evaluation ({})'.format(preds_data_type)):

            # Get clean, noisy/enhanced audio dir
            clean_sp_file_dir = os.path.join(target_dir, file)
            enhanced_sp_file_dir = os.path.join(preds_dir, filename_prefix + file)

            # Load target, preds audio
            clean_sp = self.dataset_manager.load_audio(file_dir=clean_sp_file_dir, pad_flag=False,
                                                       update_info=False)
            enhanced_sp = self.dataset_manager.load_audio(file_dir=enhanced_sp_file_dir, pad_flag=False,
                                                          update_info=False)

            # Adjust audio length
            min_len = min(clean_sp.shape[-1], enhanced_sp.shape[-1])
            enhanced_sp = enhanced_sp.view(enhanced_sp.shape[-1])[:min_len]
            clean_sp = clean_sp.view(clean_sp.shape[-1])[:min_len]

            # Compute PESQ
            pesq_val = self.pesq_fn(preds=enhanced_sp, target=clean_sp)
            pesq_list.append(pesq_val)

            # Compute STOI
            stoi_val = self.stoi_fn(preds=enhanced_sp, target=clean_sp)
            stoi_list.append(stoi_val)

            # Compute ESTOI
            estoi_val = self.stoi_fn(preds=enhanced_sp, target=clean_sp, extended=True)
            estoi_list.append(estoi_val)

            # # Compute SI-SDR
            # si_sdr_val = self.si_sdr_fn(preds=enhanced_sp, target=clean_sp)
            # si_sdr_list.append(si_sdr_val)

            # Compute SI-SNR
            si_snr_val = self.si_snr_fn(preds=enhanced_sp, target=clean_sp)
            si_snr_list.append(si_snr_val)

        mean_pesq = np.mean(pesq_list)
        mean_stoi = np.mean(stoi_list)
        mean_estoi = np.mean(estoi_list)
        # mean_si_sdr = np.mean(si_sdr_list)
        mean_si_snr = np.mean(si_snr_list)

        self.obj_evaluation_dc[f'{preds_data_type} PESQ score'] = float(np.round(mean_pesq, self.nb_digits))
        self.obj_evaluation_dc[f'{preds_data_type} STOI score'] = float(np.round(mean_stoi, self.nb_digits))
        self.obj_evaluation_dc[f'{preds_data_type} ESTOI score'] = float(np.round(mean_estoi, self.nb_digits))
        # self.obj_evaluation_dc[f'{preds_data_type} SI-SDR score'] = float(np.round(mean_si_sdr, self.nb_digits))
        self.obj_evaluation_dc[f'{preds_data_type} SI-SNR score'] = float(np.round(mean_si_snr, self.nb_digits))

        print(f'\t - objective evaluation metrics :')
        print(f'\t\t - {preds_data_type} PESQ score = {mean_pesq:.2f}')
        print(f'\t\t - {preds_data_type} STOI score = {mean_stoi:.2f}')
        print(f'\t\t - {preds_data_type} ESTOI score = {mean_estoi:.2f}')
        # print(f'\t\t - {preds_data_type} SI-SDR score = {mean_si_sdr:.2f}')
        print(f'\t\t - {preds_data_type} SI-SNR score = {mean_si_snr:.2f}')

    @staticmethod
    def pesq_fn(preds: torch.Tensor, target: torch.Tensor, fs: int = AudioParameters.sample_rate, mode: str = 'wb'):
        """Method that computes PESQ score.

        Parameters
        ----------
        preds: torch.Tensor
            'noisy', or 'enhanced' audio.
        target: torch.Tensor
            'clean' audio.
        fs: int
            Audio sampling frequency.
        mode: str
            'wb' or 'nb'.
        """
        return pesq(fs, target.cpu().numpy(), preds.cpu().numpy(), mode)

    @staticmethod
    def stoi_fn(preds: torch.Tensor, target: torch.Tensor, fs: int = AudioParameters.sample_rate,
                extended: bool = False):
        """Method that computes STOI/ESTOI score.

        Parameters
        ----------
        preds: torch.Tensor
            'noisy', or 'enhanced' audio.
        target: torch.Tensor
            'clean' audio.
        fs: int
            Audio sampling frequency.
        extended: str
            Boolean that indicates weather to use STOI or ESTOI function.
        """
        return stoi(target.cpu().numpy(), preds.cpu().numpy(), fs, extended)

    @staticmethod
    def si_sdr_fn(preds: torch.Tensor, target: torch.Tensor):
        """Method that computes SI-SDR score.

        Parameters
        ----------
        preds: torch.Tensor
            'noisy', or 'enhanced' audio.
        target: torch.Tensor
            'clean' audio.
        """
        return ScaleInvariantSignalDistortionRatio()(preds, target)

    @staticmethod
    def si_snr_fn(preds: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8):
        """Method that computes SI-SNR score.

        Parameters
        ----------
        preds: torch.Tensor
            'noisy', or 'enhanced' audio.
        target: torch.Tensor
            'clean' audio.
        epsilon: float
            A small value to avoid computation error.
        """
        s_preds_ = preds - torch.mean(preds, dim=-1, keepdim=True)
        s_target_ = target - torch.mean(target, dim=-1, keepdim=True)
        s_preds_dot_s_target_ = torch.sum(s_preds_ * s_target_, dim=-1, keepdim=True)
        s_target_norm_ = torch.sum(s_target_ ** 2, dim=-1, keepdim=True)

        s_target = s_preds_dot_s_target_ * s_target_ / s_target_norm_
        e_noise = s_preds_ - s_target

        s_target_norm = torch.sum(s_target ** 2, dim=-1)
        e_noise_norm = torch.sum(e_noise ** 2, dim=-1) + epsilon
        si_snr = 10 * torch.log10((s_target_norm / e_noise_norm) + epsilon)

        return si_snr.item()

    def compute_composite_objective_eval_metrics(self, preds_dir: str, target_dir: str, preds_data_type: str,
                                                 filename_prefix: str = '') -> None:
        """Method computes composite metrics: CSIG, CBAK, COVL and segSNR.

        Parameters
        ----------
        preds_dir: str
            Noisy/Clean/Enhanced audio data directory.
        target_dir: str
            Clean audio data directory.
        preds_data_type: str
            'noisy' or 'enhanced' audio.
        filename_prefix: bool
            Enhanced audio data filename prefix.
        """

        list_speech_audio = sorted(os.listdir(preds_dir))
        if len(filename_prefix) > 0:
            list_speech_audio = [list_speech_audio[i][len(filename_prefix):] for i in range(len(list_speech_audio))]

        csig_list = []
        cbak_list = []
        covl_list = []
        ssnr_list = []
        for file in tqdm(list_speech_audio, leave=True, desc='Composite Objective Evaluation ({})'.format(preds_data_type)):

            # Get clean, noisy/enhanced audio dir
            clean_sp_file_dir = os.path.join(target_dir, file)
            enhanced_sp_file_dir = os.path.join(preds_dir, filename_prefix + file)

            # Load target, preds audio
            clean_sp = self.dataset_manager.load_audio(file_dir=clean_sp_file_dir, pad_flag=False,
                                                       update_info=False)
            enhanced_sp = self.dataset_manager.load_audio(file_dir=enhanced_sp_file_dir, pad_flag=False,
                                                          update_info=False)

            # Adjust audio length
            min_len = min(clean_sp.shape[-1], enhanced_sp.shape[-1])
            enhanced_sp = enhanced_sp.view(enhanced_sp.shape[-1])[:min_len]
            clean_sp = clean_sp.view(clean_sp.shape[-1])[:min_len]

            # Compute Csig, Cbak, Covl
            csig_val, cbak_val, covl_val, _, ssnr_val = CompositeEval(clean_sp.view(-1).numpy(),
                                                                      enhanced_sp.view(-1).numpy(),
                                                                      log_all=True)
            csig_list.append(csig_val)
            cbak_list.append(cbak_val)
            covl_list.append(covl_val)
            ssnr_list.append(ssnr_val)

        mean_csig = np.mean(csig_list)
        mean_cbak = np.mean(cbak_list)
        mean_covl = np.mean(covl_list)
        mean_ssnr = np.mean(ssnr_list)

        self.composite_obj_evaluation_dc[f'{preds_data_type} Csig score'] = float(np.round(mean_csig, self.nb_digits))
        self.composite_obj_evaluation_dc[f'{preds_data_type} Cbak score'] = float(np.round(mean_cbak, self.nb_digits))
        self.composite_obj_evaluation_dc[f'{preds_data_type} Covl score'] = float(np.round(mean_covl, self.nb_digits))
        self.composite_obj_evaluation_dc[f'{preds_data_type} segSNR score'] = float(np.round(mean_ssnr, self.nb_digits))

        print(f'\t - Composite objective evaluation metrics :')
        print(f'\t\t - {preds_data_type} Csig score = {mean_csig:.2f}')
        print(f'\t\t - {preds_data_type} Cbak score = {mean_cbak:.2f}')
        print(f'\t\t - {preds_data_type} Covl score = {mean_covl:.2f}')
        print(f'\t\t - {preds_data_type} segSNR score = {mean_ssnr:.2f}')

    def compute_dnsmos_objective_eval_metrics(self, speech_dir: str, data_type: str, personalized_mos: bool = False) \
            -> None:
        """Method that computes DNSMOS scores: SIG, BAK, OVRL.

        Parameters
        ----------
        speech_dir: str
            Noisy/Clean/Enhanced audio data directory.
        data_type: str
            'noisy', 'clean' or 'enhanced' audio.
        personalized_mos: bool
            Boolean that indicate weather personalized MOS score is needed or regular.
        """

        if personalized_mos:
            mos_type = 'personalized'
        else:
            mos_type = 'regular'

        self.compute_dnsmos_score(testset_dir=speech_dir,
                                  data_type=data_type,
                                  csv_path=os.path.join(self.csv_files_dir,
                                                        '{}_{}_MOS.csv'.format(data_type, mos_type)),
                                  personalized_MOS=personalized_mos)

    def compute_dnsmos_score(self, testset_dir: str, data_type: str, csv_path: str, personalized_MOS: bool,
                             SAMPLING_RATE: int = 16000) -> None:
        """Method that computes DNSMOS scores: SIG, BAK, OVRL.

        Parameters
        ----------
        testset_dir: str
            Noisy/Clean/Enhanced audio data directory.
        data_type: str
            'noisy', 'clean' or 'enhanced' audio.
        csv_path: bool
            Output csv file saving directory.
        personalized_mos: bool
            Boolean that indicate weather personalized MOS score is needed or regular.
        SAMPLING_RATE: bool
            Audio data sampling rate.
        """

        models = glob.glob(os.path.join(testset_dir, "*"))

        audio_clips_list = []

        if personalized_MOS:
            primary_model_path = os.path.join(self.dnsmos_model_dir, 'pDNSMOS', 'sig_bak_ovr.onnx')
        else:
            primary_model_path = os.path.join(self.dnsmos_model_dir, 'DNSMOS', 'sig_bak_ovr.onnx')
        p808_model_path = os.path.join(self.dnsmos_model_dir, 'DNSMOS', 'model_v8.onnx')
        dnsmos = DNSMOS(primary_model_path, p808_model_path)

        rows = []
        clips = []
        clips = glob.glob(os.path.join(testset_dir, "*.wav"))
        is_personalized_eval = personalized_MOS
        desired_fs = SAMPLING_RATE

        for m in tqdm(models, leave=False, desc='DNSMOS Objective Evaluation ({})'.format(data_type)):
            max_recursion_depth = 10
            audio_path = os.path.join(testset_dir, m)
            audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
            while len(audio_clips_list) == 0 and max_recursion_depth > 0:
                audio_path = os.path.join(audio_path, "**")
                audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
                max_recursion_depth -= 1
            clips.extend(audio_clips_list)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_url = {executor.submit(dnsmos, clip, desired_fs, is_personalized_eval): clip for clip in
                             clips}
            for future in tqdm(concurrent.futures.as_completed(future_to_url)):
                clip = future_to_url[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (clip, exc))
                else:
                    rows.append(data)

        df = pd.DataFrame(rows)
        if csv_path:
            csv_path = csv_path
            df.to_csv(csv_path)
        else:
            print(df.describe())

        self.compute_mean_dnsmos(rows=rows, data_type=data_type)

    def compute_mean_dnsmos(self, rows: list, data_type: str) -> None:
        """Method that computes DNSMOS scores: SIG, BAK, OVRL.

        Parameters
        ----------
        rows: list
            List of audio data.
        data_type: str
            'noisy', 'clean' or 'enhanced' audio.
        """

        dnsmos_dict = {'OVRL_raw': 0., 'SIG_raw': 0., 'BAK_raw': 0., 'OVRL': 0., 'SIG': 0., 'BAK': 0., 'P808_MOS': 0.}

        for i in range(len(rows)):
            for dnsmos_name in dnsmos_dict:
                dnsmos_dict[dnsmos_name] += rows[i][dnsmos_name]

        print(f'\t - DNSMOS objective evaluation metrics :')
        for dnsmos_name in dnsmos_dict:
            dnsmos_mean = dnsmos_dict[dnsmos_name] / len(rows)
            self.dnsmos_obj_evaluation_dc[f'{data_type} {dnsmos_name} score'] = float(np.round(dnsmos_mean, self.nb_digits))
            if dnsmos_name in ['OVRL', 'SIG', 'BAK', 'P808_MOS']:
                print(f'\t\t - {data_type} {dnsmos_name} score = {dnsmos_mean:.2f}')


