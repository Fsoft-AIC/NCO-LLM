import torch

def heuristics2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    total_demand = demands.sum().item()
    demand_norm = demands / total_demand
    edge_savings = distance_matrix - demand_norm[:, None] - demand_norm
    return edge_savings
