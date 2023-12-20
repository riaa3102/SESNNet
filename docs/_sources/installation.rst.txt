Quickstart
============

Installation
----------------

Install the packages using pip:

.. code-block:: console

   $ pip install -r requirements.txt

Usage
-----

To train a speech enhancement SNN model, run the following command:

.. code-block:: console

   python main.py --model_name UNetSNN --train_flag --nb_epochs 5 --batch_size 4 --learning_rate 0.0004 --train_neuron_parameters --recurrent_flag --detach_reset --use_ddp --pin_memory --empty_cache --evaluate_flag --perceptual_metric_flag --save_mem --deterministic

To test a pretrained speech enhancement SNN model, run the following command:

.. code-block:: console

   python main.py --model_name UNetSNN --pretrained_flag --batch_size 4 --train_neuron_parameters --recurrent_flag --detach_reset --use_ddp --pin_memory --empty_cache --evaluate_flag --perceptual_metric_flag --save_mem --deterministic

.. note::
    If input data (for example STFT representation) is not computed, please add the following argument :

        --compute_representation

.. note::
    In order to use debug mode, please add the following argument :

        --debug_flag

Arguments
---------

These are arguments that can be set from the command line args.

.. argparse::
    :filename: ../main.py
    :func: argument_parser
    :prog: python

Experiment tracking using Comet ML
-----------------------------------

Sign up on `comet.ml <https://www.comet.com/site/>`_, and add the following arguments to command line:

.. code-block:: console

   --workspace <Your Workspace> --api_key <Your API Key> --project_name <Your Project Name>

