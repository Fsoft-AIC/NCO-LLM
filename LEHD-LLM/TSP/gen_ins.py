import numpy as np
import os
np.random.seed(1234)

def generate_tsp_instances(num_instances, num_nodes):
    # Ensure the datasets directory exists
    if not os.path.exists("datasets"):
        os.mkdir("datasets")
    
    # Open a single file for all instances
    filename = f"datasets/all_TSP_{num_nodes}_instances.txt"
    with open(filename, "w") as file:
        # Generate and save each instance
        for i in range(num_instances):
            # Generate num_nodes random 2D points
            nodes = np.random.rand(num_nodes, 2)
            
            # Format the instance as a single line
            instance_line = ' '.join([f"{node[0]} {node[1]}" for node in nodes])
            
            # Write the instance to the file, each on a new line
            file.write(instance_line + "\n")

# Example usage
generate_tsp_instances(2, 10)  # Generates 2 instances with 10 nodes each, saved in one file