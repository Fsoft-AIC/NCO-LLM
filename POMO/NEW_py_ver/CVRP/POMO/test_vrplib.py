import vrplib
import numpy as np
import torch
import yaml
import json
import time
import os
from torch.optim import Adam as Optimizer

DEBUG_MODE = False
USE_CUDA = True
CUDA_DEVICE_NUM = 0

# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
from logging import getLogger
from utils.utils import *


# from CVRPModel import CVRPModel, CVRPModel_local
# from CVRPEnv import CVRPEnv
# from utils import rollout, check_feasible

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model
# from CVRPModel_lm import CVRPModel as Model

logger_params = {
    'log_file': {
        'desc': 'test_cvrp100_pomo-lm',
        'filename': 'log.txt'
    }
}

env_params = {
    'problem_size': 100,
    'pomo_size': 100,
    'vrplib_set': 'XXL',
    'vrplib': True,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
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
        'path': './result/saved_CVRP100_model',  # directory path of pre-trained model and log files saved.
        'epoch': 30500,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 10,
    'test_batch_size': 10,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 100,
    'test_data_load': {
        'enable': False,
        'filename': './vrp100_test_seed1234.pt'
    },
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']




class VRPLib_Tester:

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
        self.result_folder = get_result_folder()
        
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
        
        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # load trained model
        # self.model = Model(**model_params)
        # if model_params['ensemble']:
        #     self.model.decoder.add_local_policy(self.device)

        # checkpoint = torch.load(load_checkpoint, map_location=self.device)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        self.vrplib_path = 'VRPLib/Vrp-Set-X/' if env_params['vrplib_set'] == 'X' else "VRPLib/Vrp-Set-XXL/"
        self.repeat_times = 1
        self.aug_factor = self.tester_params['aug_factor'] 
        self.vrplib_results = None
        
    def test_on_vrplib(self):
        files = os.listdir(self.vrplib_path)
        vrplib_results = []
        total_time = 0.
        for t in range(self.repeat_times):
            for name in files:
                if '.sol' in name:
                    continue
                name = name[:-4]
                instance_file = self.vrplib_path + '/' + name + '.vrp'
                solution_file = self.vrplib_path + '/' + name + '.sol'
                
                solution = vrplib.read_solution(solution_file)
                optimal = solution['cost']

                result_dict = {}
                result_dict['run_idx'] = t
                start_time = time.time()
                self.test_on_one_ins(name=name, result_dict=result_dict, instance=instance_file, solution=solution_file)
                total_time += time.time() - start_time

                new_instance_dict = {}
                new_instance_dict['instance'] = name
                new_instance_dict['optimal'] = optimal
                new_instance_dict['record'] = [result_dict]
                vrplib_results.append(new_instance_dict)

                print("Instance Name {}: gap {:.4f} no_aug {:.4f}".format(name, result_dict['gap'], result_dict['gap_noaug']))
                if 'XXL' in self.vrplib_path:
                    print("gap {:.4f} no_aug {:.4f}".format(result_dict['gap'], result_dict['gap_noaug']))
        if 'XXL' in self.vrplib_path:
            avg_gap = []
            noaug_gap = []
            for result in vrplib_results:
                avg_gap.append(result['record'][-1]['gap'])
                noaug_gap.append(result['record'][-1]['gap_noaug'])
            
            print("{:.3f}%".format(100 * np.array(avg_gap).mean()))
            print("{:.3f}%".format(100 * np.array(noaug_gap).mean()))
            print("Average time: {:.2f}s".format(total_time / 4))
        else:
            total = []
            number = 0
            avg_very_small_gap = []
            avg_small_gap = []
            avg_medium_gap = []
            avg_large_gap = []
            avg_very_large_gap = []
            
            for result in vrplib_results:
                scale = int(result['record'][-1]['scale'])
                # if scale <= 2000:
                total.append(result['record'][-1]['gap'])
                if scale <100: 
                    avg_very_small_gap.append(result['record'][-1]['gap'])
                if 100 <= scale < 200:
                    avg_small_gap.append(result['record'][-1]['gap'])
                elif 200<= scale < 500:
                    avg_medium_gap.append(result['record'][-1]['gap'])
                elif 500<= scale <= 1000:
                    avg_large_gap.append(result['record'][-1]['gap'])
                elif scale >1000 :
                    avg_very_large_gap.append(result['record'][-1]['gap'])
                number += 1
                
            print("Average gap on subset of <200: {:.3f}%".format(100 * np.array(avg_small_gap).mean()))
            print("Average gap on subset of 200-500: {:.3f}%".format(100 * np.array(avg_medium_gap).mean()))
            print("Average gap on subset of 500-1000: {:.3f}%".format(100 * np.array(avg_large_gap).mean()))
            print("Average gap total: {:.3f}%".format(100 *(np.array(total).mean())))
            print("Average time: {:.3f}s".format(total_time / number))
            
            vrplib_results.append({"<200": 100 * np.array(avg_small_gap).mean(), 
            "200-500": 100 * np.array(avg_medium_gap).mean(),
            "500-1000": 100 * np.array(avg_large_gap).mean(),
            "total": 100 *(np.array(total).mean())})
            with open('result/' + self.config['name'] + '_' + 'vrplib.json', 'w') as f:
                json.dump(vrplib_results, f)


    def test_on_one_ins(self, name, result_dict, instance, solution):
        instance = vrplib.read_instance(instance)
        solution = vrplib.read_solution(solution)
        optimal = solution['cost']
        problem_size = instance['node_coord'].shape[0] - 1
        multiple_width = min(problem_size, 1000)
        
        # multiple_width = problem_size
        self.env_params['problem_size'] = problem_size
        self.env_params['pomo_size'] = multiple_width
        
        # initialize env
        self.env = Env(**self.env_params)
        # reset_state, reward, done = self.env.reset()
        
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1


        # Initialize CVRP state
        # env = Env(multiple_width, self.device)
        # self.env.load_vrplib_problem(instance, aug_factor=self.aug_factor)

        # reset_state, reward, done = env.reset()
        # self.model.eval()
        # self.model.requires_grad_(False)
        # self.model.pre_forward(reset_state)
        # Ready
        ###############################################
        self.model.eval()
        
        with torch.no_grad():
            self.env.load_vrplib_problem(instance, aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # with torch.no_grad():
        #     policy_solutions, policy_prob, rewards = rollout(self.model, env, 'greedy')
        # # Return
        
        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
        
        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, 1, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        # return no_aug_score.item(), aug_score.item()

        if result_dict is not None:
            result_dict['best_cost'] = aug_score.item()
            result_dict['noaug_cost'] = no_aug_score.item()
            result_dict['scale'] = problem_size
            result_dict['gap'] = (result_dict['best_cost'] - optimal) / optimal
            result_dict['gap_noaug'] = (result_dict['noaug_cost'] - optimal) / optimal
            # print(best_cost)


if __name__ == "__main__":
    tester = VRPLib_Tester(env_params=env_params,
                        model_params=model_params,
                        tester_params=tester_params)
    tester.test_on_vrplib()