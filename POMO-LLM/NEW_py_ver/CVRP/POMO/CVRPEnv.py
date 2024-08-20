
from dataclasses import dataclass
import torch
from tqdm import tqdm

from CVRProblemDef import get_random_problems, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class CVRPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.vrplib = env_params['vrplib']
        self.device = env_params['device']
        if env_params['test_file_path']: self.test_file_path = env_params['test_file_path']
        if env_params['n_samples']: self.n_samples = env_params['n_samples']
        # if env_params['device']: self.device = env_params['device']
        # if env_params['lehd']:
        #     self.depot_xy_all = [] # (n_samples, 1, 2)
        #     self.node_xy_all = [] # (n_samples, problem, 2)
        #     self.node_demand_all = [] # (n_samples, problem)
        #     if self.problem_size == 20:
        #         self.demand_scaler = 30
        #     elif self.problem_size == 50:
        #         self.demand_scaler = 40
        #     elif self.problem_size == 100:
        #         self.demand_scaler = 50
        #     elif self.problem_size == 150:
        #         self.demand_scaler = 60
        #     elif self.problem_size == 200:
        #         self.demand_scaler = 80
        #     elif self.problem_size == 500:
        #         self.demand_scaler = 100
        #     elif self.problem_size == 1000:
        #         self.demand_scaler = 250
        #     elif self.problem_size == 5000:
        #         self.demand_scaler = 500
        #     else:
        #         raise NotImplementedError
        

            

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']
        self.saved_index = 0

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size


        if not self.FLAG__use_saved_problems:
            depot_xy, node_xy, node_demand = get_random_problems(batch_size, self.problem_size)
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
            else:
                raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)


        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        
    def load_problems_lehd(self, episode, batch_size, aug_factor=1):
        # def tow_col_nodeflag(node_flag):
        #     tow_col_node_flag = []
        #     V = int(len(node_flag) / 2)
        #     for i in range(V):
        #         tow_col_node_flag.append([node_flag[i], node_flag[V + i]])
        #     return tow_col_node_flag
        
        
        self.batch_size = batch_size
        self.episode = episode
        if self.test_file_path is None:
            raise NotImplementedError
        # print('load raw dataset begin!')
        # raw_data_tours = []
        if episode==0:

            for line in tqdm(open(self.test_file_path, "r").readlines()[0:self.n_samples], ascii=True):
                line = line.split(",")

                depot_index = int(line.index('depot'))
                customer_index = int(line.index('customer'))
                capacity_index = int(line.index('capacity'))
                demand_index = int(line.index('demand'))
                cost_index = int(line.index('cost'))
                node_flag_index = int(line.index('node_flag'))

                depot = [float(line[depot_index + 1]), float(line[depot_index + 2])]
                customer = [[float(line[idx]), float(line[idx + 1])] for idx in range(customer_index + 1, capacity_index, 2)]

                # loc = depot + customer
                self.depot_xy_all.append(depot)
                self.node_xy_all.append(customer)
                # capacity = int(float(line[capacity_index + 1])) 
                if int(line[demand_index + 1]) ==0:
                    demand = [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
                else:
                    demand = [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
                    
                self.node_demand_all.append(demand)
                # cost = float(line[cost_index + 1])
                # node_flag = [int(line[idx]) for idx in range(node_flag_index + 1, len(line))]

                # node_flag = tow_col_nodeflag(node_flag)

        depot_xy = self.depot_xy_all[episode:episode+batch_size]
        node_xy = self.node_xy_all[episode:episode+batch_size]
        node_demand = self.node_demand_all[episode:episode+batch_size]
        depot_xy = torch.tensor(depot_xy).unsqueeze(1)
        node_xy = torch.tensor(node_xy)
        node_demand = torch.tensor(node_demand)/float(self.demand_scaler)

            # shape (B,V+1,2)  customer num + depot
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
            else:
                raise NotImplementedError
            
        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)


        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        
    def load_vrplib_problem(self, instance, aug_factor=1):
        self.vrplib = True
        self.batch_size = 1
        node_coord = torch.FloatTensor(instance['node_coord']).unsqueeze(0).to(self.device)
        demand = torch.FloatTensor(instance['demand']).unsqueeze(0).to(self.device)
        demand = demand / instance['capacity']
        self.unscaled_depot_node_xy = node_coord
        # shape: (batch, problem+1, 2)
        
        min_x = torch.min(node_coord[:, :, 0], 1)[0]
        min_y = torch.min(node_coord[:, :, 1], 1)[0]
        max_x = torch.max(node_coord[:, :, 0], 1)[0]
        max_y = torch.max(node_coord[:, :, 1], 1)[0]
        scaled_depot_node_x = (node_coord[:, :, 0] - min_x) / (max_x - min_x)
        scaled_depot_node_y = (node_coord[:, :, 1] - min_y) / (max_y - min_y)
        
        # self.depot_node_xy = self.unscaled_depot_node_xy / 1000
        self.depot_node_xy = torch.cat((scaled_depot_node_x[:, :, None]
                                        , scaled_depot_node_y[:, :, None]), dim=2)
        depot = self.depot_node_xy[:, instance['depot'], :]
        # shape: (batch, problem+1)
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot = augment_xy_data_by_8_fold(depot)
                self.depot_node_xy = augment_xy_data_by_8_fold(self.depot_node_xy)
                self.unscaled_depot_node_xy = augment_xy_data_by_8_fold(self.unscaled_depot_node_xy)
                demand = demand.repeat(8, 1)
            else:
                raise NotImplementedError
        
        self.depot_node_demand = demand
        self.reset_state.depot_xy = depot
        self.reset_state.node_xy = self.depot_node_xy[:, 1:, :]
        self.reset_state.node_demand = demand[:, 1:]
        self.problem_size = self.reset_state.node_xy.shape[1]

        self.dist = (self.depot_node_xy[:, :, None, :] - self.depot_node_xy[:, None, :, :]).norm(p=2, dim=-1)
        # shape: (batch, problem+1, problem+1)
        self.reset_state.dist = self.dist
        

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX


    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1 # refill loaded at the depot

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, problem+1)

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            if self.vrplib == True:
                reward = self.compute_unscaled_reward()
            else:
                reward = -self._get_travel_distance()
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def compute_unscaled_reward(self, solutions=None, rounding=True):
        if solutions is None:
            solutions = self.selected_node_list
        gathering_index = solutions[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, multi, selected_list_length, 2)
        all_xy = self.unscaled_depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, multi, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, multi, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)

        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        if rounding == True:
            segment_lengths = torch.round(segment_lengths)
        # shape: (batch, multi, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, multi)
        return -travel_distances

