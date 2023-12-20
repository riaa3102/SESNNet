"""
File:
    data/constants.py

Description:
    This file stores helpful constants value.
"""

from pathlib import Path


# class DataAttributes(Enum):
#     # Define device as the GPU if available, otherwise use the CPU
#     # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     if torch.cuda.is_available():
#         # device = torch.device('cuda')
#         # device = torch.device(f'cuda:{dist.get_rank()}')
#         ngpus_per_node = torch.cuda.device_count()
#         device_ids = list(range(torch.cuda.device_count()))
#         # gpus = len(device_ids)
#
#     # define data_files type
#     dtype = torch.float


class DataDirectories(str):
    # Project directory
    project_dir = Path(__file__).parent.parent.parent

    # Set loaded data type
    data_load_train = 'train'
    data_load_valid = 'valid'
    data_load_test = 'test'

    # Set data dir names
    data_dirname = 'data'
    audio_dirname = 'audio'

    coefficients_dirname = 'coefficients'
    lca_weights_dirname = 'weights'
    lca_weights_filename = 'weights'

    audio_info_dirname = '__audio_info__'
    transform_info_dirname = 'metadata'
    transform_info_filename = 'metadata'

    noisyspeech_dirname = 'noisy'
    cleanspeech_dirname = 'clean'

    enhancedspeech_dirname = 'enhanced'
    coefficients_filename = 'coefficients_'
    enhancedspeech_filename = 'enh'

    trained_models_dirname = 'trained_models'
    experiments_dirname = 'experiments'


class AudioParameters:
    audio_type: str = 'wav'
    sample_rate: int = 16000
    # Maximum duration of audio files to consider (seconds)
    max_duration: float = 4

class RepresentationName:
    lca = 'lca'
    stft = 'stft'
