
from dataclasses import dataclass
import torch

from TSProblemDef import get_random_problems, augment_xy_data_by_8_fold
from tqdm import tqdm


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)


class TSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        # self.test_file_path = env_params['test_file_path']
        
        #### if using tsplib to test: please comment these code, 
        
        self.tsplib = env_params['tsplib']
        if self.tsplib == False:
            if env_params['test_file_path'] is not None: self.test_file_path = env_params['test_file_path']
            if env_params['n_samples']: self.n_samples = env_params['n_samples']
            if env_params['device']: self.device = env_params['device']
            if env_params['lehd']:
                self.raw_data_nodes = []
        
            
        

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size
        # print(self.test_file_path)
        if self.test_file_path is not None:
            self.problems = torch.load(self.test_file_path).to(self.device)
        else:
            self.problems = get_random_problems(batch_size, self.problem_size)
        # problems.shape: (batch, problem, 2)
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)
                # shape: (8*batch, problem, 2)
            else:
                raise NotImplementedError

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
    
    
    def load_problems_lehd(self, episode, batch_size, aug_factor=1):
        self.batch_size = batch_size
        if self.test_file_path is None:
            raise NotImplementedError
        # print('load raw dataset begin!')
        # raw_data_tours = []
        if episode==0:
            for line in tqdm(open(self.test_file_path, "r").readlines()[0:self.n_samples], ascii=True):
                line = line.split(" ")
                num_nodes = int(line.index('output') // 2)
                nodes = [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]

                self.raw_data_nodes.append(nodes)
                # tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]]

                # raw_data_tours.append(tour_nodes)

            self.raw_data_nodes = torch.tensor(self.raw_data_nodes,requires_grad=False)
        # print(f'load raw dataset done!', )

        self.problems = self.raw_data_nodes[episode:episode + batch_size]
        # shape: [B,V,2]  ;  shape: [B,V]

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)
                # shape: (8*batch, problem, 2)
            else:
                raise NotImplementedError
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
 


    def load_tsplib_problem(self, problems, unscaled_problems, aug_factor=1):
        self.tsplib = True
        self.batch_size = problems.size(0)
        self.problem_size = problems.size(1)
        self.problems = problems
        self.unscaled_problems = unscaled_problems
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)
                # shape: (8*batch, problem, 2)
            else:
                raise NotImplementedError

        self.dist = (self.problems[:, :, None, :] - self.problems[:, None, :, :]).norm(p=2, dim=-1) # (batch, problem, problem)
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)  # (batch_size, pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        # shape: (batch, pomo, problem)

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)

        # returning values
        done = (self.selected_count == self.problem_size)
        if done:
            if self.tsplib == True:
                reward = self.compute_unscaled_distance()
            else:
                
                reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done
    

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances
    
    def compute_unscaled_distance(self, solutions=None):
        if solutions is None:
            solutions = self.selected_node_list
        multi_width = solutions.shape[1]
        # Gather instance in order of tour
        d = self.unscaled_problems[:, None, :, :].expand(self.batch_size, multi_width, self.problem_size, 2)\
            .gather(2, solutions[:, :, :, None].expand(self.batch_size, multi_width, self.problem_size, 2))
        # shape: (batch, multi, problem, 2)

        rolled_seq = d.roll(dims=2, shifts=-1)
        return -torch.round(((d-rolled_seq)**2).sum(3).sqrt()).sum(2)

