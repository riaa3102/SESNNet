Project Files Structure
========================

Diagram
-------

The project's directory structure is the following:

.. code-block:: text

    Speech_enhancement_SNN/
    ├── app.py
    ├── constants.py
    ├── experiments/
    ├── main.py
    ├── requirements.txt
    ├── comet_ml_params.txt
    ├── SpeechEnhancement_data/
    │   ├── audio/
    │   │   ├── __audio_info__/
    │   │   ├── clean_test/
    │   │   ├── clean_train/
    │   │   ├── clean_valid/
    │   │   ├── noisy_test/
    │   │   ├── noisy_train/
    │   │   └── noisy_valid/
    │   └── STFT_4s_nfft=512_wl=512_hl=256/
    │       ├── coefficients/
    │       ├── metadata/
    │       └── reconstruction/
    ├── src/
    │   ├── data/
    │   │   ├── __pycache__/
    │   │   ├── constants.py
    │   │   ├── DatasetManager.py
    │   │   └── TransformManager.py
    │   ├── evaluation/
    │   │   ├── __pycache__/
    │   │   ├── composite.py
    │   │   ├── DNSMOS/
    │   │   └── EvaluationManager.py
    │   ├── model/
    │   │   ├── __pycache__/
    │   │   ├── ArtificialBlock.py
    │   │   ├── ArtificialModel.py
    │   │   ├── constants.py
    │   │   ├── LossManager.py
    │   │   ├── SpeechEnhancer.py
    │   │   ├── SpikingBlock.py
    │   │   ├── SpikingLayer.py
    │   │   ├── SpikingModel.py
    │   │   ├── SurrogateGradient.py
    │   │   ├── TrainValidTestManager.py
    │   │   └── utils.py
    │   ├── stft/
    │   │   ├── __pycache__/
    │   │   ├── constants.py
    │   │   ├── Stft.py
    │   │   └── StftManager.py
    │   └── visualization/
    │       ├── __pycache__/
    │       └── VisualizationManager.py
    ├── docsrc/
    │   ├── make.bat
    │   ├── Makefile
    │   └── source/
    ├── docs/
    └── trained_models/

Description
-----------

The ``Speech_enhancement_SNN`` directory is the project's root directory. It holds the following folders:

* ``experiments``: This folder contains the output data, generalized figures, and experiment parameters and results.

* ``SpeechEnhancement_data``: This folder contains the data used in the project, including the following subfolders:

    * ``audio``: This subfolder contains audio files used for training and testing the models, divided into subfolders for clean and noisy data for both training and testing, as well as a subfolder for audio information.
    * ``STFT_4s_nfft=512_wl=512_hl=256``: This subfolder contains short-time Fourier transform (STFT) coefficients, metadata, and reconstructed audio data for a subset of the processed audio files.

* ``src``: This folder contains the project's source code, including the following subfolders:

    * ``data``: This subfolder contains constants, the dataset manager, and the transform manager used to preprocess the audio data.
    * ``evaluation``: This subfolder contains code related to evaluating the performance of the models using objective metrics.
    * ``model``: This subfolder contains the code for the models used in the project, including artificial and spiking models, traning and testing class, loss functions, surrogate gradients, and the speech enhancer main class.
    * ``stft``: This subfolder contains the code for computing the STFT.
    * ``visualization``: This subfolder contains code related to visualizing the audio data, loss curve and the models' input and output data.

* ``docsrc``: This subfolder contains the documentation for the project.

* ``trained_models``: This folder contains the trained models.
