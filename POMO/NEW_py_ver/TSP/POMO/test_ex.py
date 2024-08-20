##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = True
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

from TSPTester import TSPTester as Tester
from gen_inst import generate_datasets

data_params = {
    # problem_size: [episodes, batch]
    100: [10000, 2000],
    200: [128, 128],
    500: [128, 128],
    1000: [128, 32],
}

##########################################################################################
# parameters

logger_params = {
    'log_file': {
        'desc': 'test_tsp100_pomo',
        'filename': 'log.txt'
    }
}
env_params = {
    'problem_size': 100,
    'pomo_size': 100,
    'tsplib': False,
    
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'decoder_layer_num':1,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 50,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}


tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': 'checkpoint of fine-tuned model here',  # directory path of pre-trained model and log files saved.
        'epoch': 800,  # epoch version of pre-trained model to load.
    },
    'test_episodes': 64,
    'test_batch_size': 64,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 100,
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']



##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()
    
    basepath = os.path.dirname(__file__)
    datas = ['test_TSP100_n1w', 'test_TSP200_n128', 'test_TSP500_n128', 'test_TSP1000_n128']
    opt = [7.763244, 10.703612, 16.521471, 23.119862]
    # opt = [10.703612, 16.521471, 23.119862]
    
    for i, problem_size in zip(range(0,len(data_params)), data_params.keys()):
        if problem_size < 100:
            tester_params['use_cuda'] = False
        env_params['problem_size'] = problem_size
        n_samples = data_params[problem_size][0]
        # if not os.path.isfile(os.path.join(basepath, f"dataset/test{problem_size}_n{n_samples}.pt")):
        #     generate_datasets(data_params)


        # dataset_path = os.path.join(basepath, f"dataset/test{problem_size}_n{n_samples}.pt")
        dataset_path = '/data' + datas[i] +'.txt' # testing data folder
        env_params['test_file_path'] = dataset_path
        env_params['problem_size'] = problem_size
        env_params['n_samples'] = n_samples
        env_params['lehd'] = True
        tester_params['test_episodes'] = n_samples
        tester_params['test_batch_size'] = data_params[problem_size][1]

        tester = Tester(env_params=env_params,
                        model_params=model_params,
                        tester_params=tester_params)
        avg_obj, avg_aug_obj = tester.run()
        gap = (avg_obj-opt[i])/opt[i]*100
        gap_aug =  (avg_aug_obj-opt[i])/opt[i]*100
        print(f"====== Average score for {problem_size}:=======")
        print(" NO-AUG SCORE and GAP: {:.4f}, {:.4f} ".format(avg_obj, gap))
        print(" AUGMENTATION SCORE and GAP: {:.4f}, {:.4f} ".format(avg_aug_obj, gap_aug))


    


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":

    main()
