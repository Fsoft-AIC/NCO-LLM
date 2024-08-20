
import torch
import numpy as np


def get_random_problems(batch_size, problem_size):
    problems = torch.rand(size=(batch_size, problem_size, 2))
    # problems.shape: (batch, problem, 2)
    return problems
def get_distance_matrix(problems):
    # problems.shape: (batch, problem, 2)
    batch_size, problem_size, _ = problems.size()
    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    x_t = x.transpose(1, 2)
    y_t = y.transpose(1, 2)
    # x_t, y_t shape: (batch, 1, problem)

    x_tile = x_t.repeat(1, problem_size, 1)
    y_tile = y_t.repeat(1, problem_size, 1)
    # x_tile, y_tile shape: (batch, problem, problem)

    x_diff = x_tile - x
    y_diff = y_tile - y
    # x_diff, y_diff shape: (batch, problem, problem)

    distance_matrix = torch.sqrt(x_diff ** 2 + y_diff ** 2)
    # distance_matrix shape: (batch, problem, problem)

    return distance_matrix

def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems

#testing
# problems = get_random_problems(1, 3)
# print(problems)
# distance_matrix = get_distance_matrix(problems)
# print(distance_matrix)