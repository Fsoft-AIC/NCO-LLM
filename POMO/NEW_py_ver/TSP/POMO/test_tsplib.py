import os
import yaml
import time
import pickle
import json
import torch
import numpy as np
from torch.optim import Adam as Optimizer

DEBUG_MODE = False
USE_CUDA = True
CUDA_DEVICE_NUM = 6

##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

from logging import getLogger
from utils.utils import *

# from generate_data import generate_tsp_data, TSPDataset
from TSPEnv import TSPEnv as Env
from TSPModel_lm import TSPModel as Model



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
    'test_path_file': None,
    'tsplib': True,
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
        'path': '/cm/shared/ML4CO/results/20240813_012352_train_tsp_n100_pomo-finetune_normalize',  # directory path of pre-trained model and log files saved.
        'epoch': 730,  # epoch version of pre-trained model to load.
    },
    'test_episodes': 64,
    'test_batch_size': 64,
    'augmentation_enable': False,
    'aug_factor': 8,
    'aug_batch_size': 100,
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']



class TSPLib_Tester:

    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):
        
        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        
        # result folder, logger
        self.logger = getLogger(name='trainer')
        # self.result_folder = get_result_folder()
        
        # self.config = config
        # model_params = config['model_params']
        # load_checkpoint = config['load_checkpoint']

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
            
        # ENV and MODEL
        env_params['device'] = self.device
        # self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)
        # print(self.model)
            
        # load trained model
        # if config['training'] == 'joint':
        #     self.model = TSPModel(**model_params)
        #     if model_params['ensemble']:
        #         self.model.decoder.add_local_policy(self.device)
        # elif config['training'] == 'only_local_att':
        #     self.model = Att_Local_policy(**model_params)
        
        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # checkpoint = torch.load(load_checkpoint, map_location=self.device)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tsplib_path = 'TSPLib_lehd'
        self.repeat_times = 1
        self.aug_factor = self.tester_params['aug_factor']    
        self.tsplib_results = None
        
    def test_on_tsplib(self):
        files = os.listdir(self.tsplib_path)
        tsplib_results = []
        total_time = 0.
        for t in range(self.repeat_times):
            for name in files:
                if '.sol' in name:
                    continue
                name = name[:-4]
                instance_file = self.tsplib_path + '/' + name + '.pkl'
                # load tsplib file
                print(instance_file)
                with open(instance_file, 'rb') as f:
                    instance = pickle.load(f)  
                    optimal = instance[1]

                result_dict = {}
                result_dict['run_idx'] = t
                start_time = time.time()
                self.test_on_one_ins(name=name, result_dict=result_dict, instance=instance)
                total_time += time.time() - start_time

                # update the results of current instance and method
                exist = False
                for result_per_instance in tsplib_results:
                    if result_per_instance['instance'] == name:
                        exist = True
                        for record in result_per_instance['record']:
                            if record['method'] == result_dict['method'] and record['run_idx'] == result_dict['run_idx']:
                                assert 'not necessary experiments!'
                                
                        result_per_instance['record'].append(result_dict)

                if exist == False:
                    new_instance_dict = {}
                    new_instance_dict['instance'] = name
                    new_instance_dict['optimal'] = optimal
                    new_instance_dict['record'] = [result_dict]
                    tsplib_results.append(new_instance_dict)

                print("Instance Name {}: gap {:.4f}".format(name, result_dict['gap']))

        with open('result/' + '_' + 'tsplib.json', 'w') as f:
            json.dump(tsplib_results, f)
        
        total_cost = []
        opt = []
        number = 0
        very_small_cost = []
        very_small_opt = []
        small_cost = []
        small_opt = []
        medium_cost = []
        medium_opt = []
        large_cost = []
        large_opt = []
        very_large_cost = []
        very_large_opt = []
        
        for result in tsplib_results:
            scale = result['record'][-1]['scale']
            # if scale <= 2000:
            opt.append(result['optimal'])
            total_cost.append(result['record'][-1]['best_cost'])
            if scale <=100: 
                very_small_cost.append(result['record'][-1]['best_cost'])
                very_small_opt.append(result['optimal'])                
            if 100 < scale <= 200:
                small_cost.append(result['record'][-1]['best_cost'])
                small_opt.append(result['optimal'])
            elif 200< scale <= 500:
                medium_cost.append(result['record'][-1]['best_cost'])
                medium_opt.append(result['optimal'])
            elif 500< scale <= 1000:
                large_cost.append(result['record'][-1]['best_cost'])
                large_opt.append(result['optimal'])
            elif scale >1000 :
                very_large_cost.append(result['record'][-1]['best_cost'])
                very_large_opt.append(result['optimal'])
            number += 1
        
        print("Total average cost {:.3f}".format(np.array(total_cost).mean()))
        print("Total average gap {:.3f}%".format(100 * ((np.array(total_cost) - np.array(opt)) / np.array(opt)).mean()))
        print("<100 average gap {:.3f}%".format(100 * ((np.array(very_small_cost) - np.array(very_small_opt)) / np.array(very_small_opt)).mean()))
        print("100-200 average gap {:.3f}%".format(100 * ((np.array(small_cost) - np.array(small_opt)) / np.array(small_opt)).mean()))
        print("200-500 average gap {:.3f}%".format(100 * ((np.array(medium_cost) - np.array(medium_opt)) / np.array(medium_opt)).mean()))
        print("500-1000 average gap {:.3f}%".format(100 * ((np.array(large_cost) - np.array(large_opt)) / np.array(large_opt)).mean()))
        print(">1000 average gap {:.3f}%".format(100 * ((np.array(very_large_cost) - np.array(very_large_opt)) / np.array(very_large_opt)).mean()))
        print("Average time: {:.3f}s".format(total_time / number))

    def test_on_one_ins(self, name, result_dict, instance):
        unscaled_points = torch.tensor(instance[0], dtype=torch.float)[None, :, :]
        points = (instance[0] - np.min(instance[0])) / (np.max(instance[0]) - np.min(instance[0]))
        # points = instance[0] / np.max(instance[0])
        test_batch = torch.tensor(points, dtype=torch.float)[None, :, :]
        optimal = instance[1]

        problem_size = test_batch.shape[1]
        # pomo_size = min(problem_size, 100)
        pomo_size = problem_size
        batch_size = test_batch.shape[0]
        
        self.env_params['problem_size'] = problem_size
        self.env_params['pomo_size'] = pomo_size
        

        # initialize env
        self.env = Env(**self.env_params)
        # reset_state, reward, done = self.env.reset()
        
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        self.model.eval()
        # self.model.requires_grad_(False)
        # self.model.pre_forward(reset_state)
        with torch.no_grad():
            # self.env.load_problems(batch_size, aug_factor)
            self.env.load_tsplib_problem(test_batch, unscaled_points, aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # policy_solutions, policy_prob, rewards = rollout(self.model, env, 'greedy')
        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)


        # aug_reward = rewards.reshape(self.aug_factor, 1, pomo_size)
        # shape: (augmentation, batch, pomo)
        
        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value


        if result_dict is not None:
            result_dict['best_cost'] = aug_score.item()
            result_dict['scale'] = problem_size
            result_dict['gap'] = (result_dict['best_cost'] - optimal) / optimal
            # print(best_cost)


if __name__ == "__main__":
    
    tester = TSPLib_Tester(env_params=env_params,
                        model_params=model_params,
                        tester_params=tester_params)
    tester.test_on_tsplib()