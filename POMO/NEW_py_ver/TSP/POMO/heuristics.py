import torch


def heuristics_v1(distance_matrix: torch.Tensor) -> torch.Tensor:
    # Prevent division by zero for self-distances
    distance_matrix[distance_matrix == 0] = 1e5
    
    # Calculating reciprocals and log transformations for the heuristic evaluation
    reciprocal = -1 / distance_matrix
    log_values = -torch.log(distance_matrix)

    # Local heuristics based on neighbor distances relative to the average
    local_mean = distance_matrix.mean(dim=1, keepdim=True)
    local_heu = log_values + 0.3 * (reciprocal.mean(dim=1, keepdim=True) - reciprocal)

    # Global characteristics based on total average distance
    global_mean = distance_matrix.mean()
    global_heu = -0.5 * torch.log(torch.abs(distance_matrix - global_mean) + 1e-5)  

    # Combine local and global heuristics into a final heuristic matrix
    heuristics_matrix = local_heu + global_heu
    
    return heuristics_matrix

def heuristics_v2(distance_matrix: torch.Tensor) -> torch.Tensor:
    """
    heuristics_v2 computes a heuristic matrix for the Traveling Salesman Problem (TSP).
    This function dynamically adjusts the radius based on clustering characteristics
    of nodes and prioritizes promising edges.

    heu_ij = 1 / (1 + dis_ij) if j is among the top K nearest neighbors of i,
    else a scaled negative distance based on clustering characteristics.
    """
    distance_matrix[distance_matrix == 0] = 1e5  # Avoid division by zero
    num_nodes = distance_matrix.size(0)
    mean_distance = torch.mean(distance_matrix)

    # Determine dynamic radius based on distance clustering
    distance_flat = distance_matrix.flatten()
    median_distance = torch.median(distance_flat[distance_flat != 1e5])
    dynamic_radius = min(median_distance.item() * 1.5, mean_distance.item() * 0.7)

    K = 10  # Using a smaller value for better efficiency
    values, indices = torch.topk(distance_matrix, k=K, largest=False, dim=1)

    heu = -distance_matrix.clone()

    # Adjust heuristic values based on cluster characteristics and dynamic radius
    for i in range(num_nodes):
        for j in range(K):
            nearest_index = indices[i, j]
            neighbor_distance = distance_matrix[i, nearest_index]
            if neighbor_distance < dynamic_radius:
                heu[i, nearest_index] = 1 / (1 + neighbor_distance)
            else:
                heu[i, nearest_index] = - neighbor_distance / mean_distance

    return heu

def heuristics_v2_optimized(distance_matrix: torch.Tensor) -> torch.Tensor:
    distance_matrix[distance_matrix == 0] = 1e5  # Avoid division by zero
    num_nodes = distance_matrix.size(0)
    mean_distance = torch.mean(distance_matrix)

    # Determine dynamic radius
    distance_flat = distance_matrix.flatten()
    median_distance = torch.median(distance_flat[distance_flat != 1e5])
    dynamic_radius = min(median_distance.item() * 1.5, mean_distance.item() * 0.7)

    K = 10
    values, indices = torch.topk(distance_matrix, k=K, largest=False, dim=1)

    # Initialize heuristic matrix with negative scaled distances
    heu = -distance_matrix / mean_distance

    # Create a mask for the top K nearest neighbors
    top_k_mask = torch.zeros_like(distance_matrix, dtype=torch.bool)
    top_k_mask.scatter_(1, indices, 1)

    # Adjust heuristic values for neighbors within the dynamic radius
    within_radius_mask = (distance_matrix < dynamic_radius) & top_k_mask
    heu[within_radius_mask] = 1 / (1 + distance_matrix[within_radius_mask])

    return heu
def heuristics_v3_optimized(distance_matrix: torch.Tensor) -> torch.Tensor:
    distance_matrix[distance_matrix == 0] = 1e5  # Avoid division by zero
    num_nodes = distance_matrix.size(0)
    mean_distance = torch.mean(distance_matrix)

    # Determine dynamic radius
    distance_flat = distance_matrix.flatten()
    median_distance = torch.median(distance_flat[distance_flat != 1e5])
    dynamic_radius = min(median_distance.item() * 1.5, mean_distance.item() * 0.7)

    K = 10
    values, indices = torch.topk(distance_matrix, k=K, largest=False, dim=1)

    # Initialize heuristic matrix with negative scaled distances
    heu = torch.full_like(distance_matrix, fill_value=-1 / mean_distance)

    # Apply heuristic for top K neighbors
    row_indices = torch.arange(num_nodes).unsqueeze(1).expand(-1, K)
    heu[row_indices, indices] = 1 / (1 + values)

    # Apply dynamic radius adjustment
    dynamic_mask = values >= dynamic_radius
    heu[row_indices[dynamic_mask], indices[dynamic_mask]] = -values[dynamic_mask] / mean_distance

    return heu
def heuristics(distance_matrix: torch.Tensor) -> torch.Tensor:
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