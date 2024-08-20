import torch

def seed_heuristics(distance_matrix: torch.Tensor, node_demands: torch.Tensor) -> torch.Tensor:
    """
    heu_ij = - log(dis_ij) if j is the topK nearest neighbor of i, else - dis_ij
    """
    distance_matrix[distance_matrix == 0] = 1e5
    K = 100
    # Compute top-k nearest neighbors (smallest distances)
    values, indices = torch.topk(distance_matrix, k=K, largest=False, dim=1)
    heu = -distance_matrix.clone()
    # Create a mask where topk indices are True and others are False
    topk_mask = torch.zeros_like(distance_matrix, dtype=torch.bool)
    topk_mask.scatter_(1, indices, True)
    # Apply -log(d_ij) only to the top-k elements
    heu[topk_mask] = -torch.log(distance_matrix[topk_mask])
    return heu
def heuristics_tsp(distance_matrix: torch.Tensor, node_demands: torch.Tensor) -> torch.Tensor:
    distance_matrix[distance_matrix == 0] = 1e5
    beta = 0.3
    gamma = 0.5
    reciprocal = -1 / distance_matrix
    log_values = -torch.log(distance_matrix)
    local_heu = log_values + beta * (reciprocal.mean(dim=1, keepdim=True) - reciprocal)
    global_mean = distance_matrix.mean()
    global_heu = -gamma * torch.log(torch.abs(distance_matrix - global_mean))
    heu = local_heu + global_heu
    return heu

def heuristics1(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    # Calculate the normalized demand-density for each edge
    norm_demand_density = 2 * demands.view(n, 1) / (distance_matrix + 1e-6) # Normalizing factor 2
    # Set penalties for edges exceeding capacity and scale the heuristics
    heuristics = norm_demand_density
    heuristics[torch.max(demands.view(n, 1), demands.view(1, n)) > 1] = -1
    return heuristics

# For CVRP500 and CVRP1000
def heuristics2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    excess_demand_penalty = torch.maximum(demands.sum() - demands, torch.tensor(0.))
    return 1 / (distance_matrix + 1e-6) - excess_demand_penalty