import torch

# dataset_conf = {
#     'train': (100, 200, 500, 1000),
#     'val':   (100, 200, 500, 1000),
#     'test':  (100, 200, 500, 1000),
# }

def generate_dataset(filepath, n, batch_size=64):
    positions = torch.rand(batch_size, n, 2)
    torch.save(positions, filepath)

def generate_datasets(data_params,basepath = None):
    import os
    basepath = basepath or os.path.join(os.path.dirname(__file__), "dataset")
    os.makedirs(basepath, exist_ok=True)

    for problem_size in data_params.keys():
        batch_size = data_params[problem_size][1]
        torch.manual_seed(1234)
        filepath = os.path.join(basepath, f"test{problem_size}_n{batch_size}.pt")
        generate_dataset(filepath, problem_size , batch_size=batch_size)


# if __name__ == "__main__":
#     generate_datasets()