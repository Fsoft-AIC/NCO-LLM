##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from TSPTrainer_ex import TSPTrainer as Trainer


##########################################################################################
# parameters

logger_params = {
    'log_file': {
        'desc': 'train_tsp_n100_pomo-finetune_normalize',
        'filename': 'log.txt'
    }
}

env_params = {
    'problem_size': 100,
    'pomo_size': 100,
    'test_file_path': None,
    'tsplib':False
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'decoder_layer_num': 1,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 50,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [510,],
        'gamma': 0.1
    }
}
extrain_params = {
    'problem_size': [100,500], # range of problem size for additional training
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 800,
    'train_episodes': 100 * 100,
    'train_batch_size': 120,
    'logging': {
        'model_save_interval': 10,
        'img_save_interval': 100,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_100.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': True,  # enable loading pre-trained model
        'path': '/cm/shared/ML4CO/results/20240802_002715_train_tsp_n100_pomo-lm-h100-v1-C50',  # directory path of pre-trained model and log files saved.
        'epoch': 510,  # epoch version of pre-trained model to laod.

    }
}



##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params,
                      extra_params=extrain_params)

    # copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 100
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
