# Large Language Models powered Neural Solvers for Generalized Vehicle Routing Problems

This repository contains code for an efficient LLM-guided fine-tuning approach to enhance the large-scale generalization of Neural Solvers for solving TSP (Traveling Salesman Problem) and CVRP (Capacitated Vehicle Routing Problem).

## Table of Contents
- [Dependencies](#dependencies)
  - [For Attention bias via LLM Design](#for-attention-bias-via-llm-design)
  - [For Fine-tuning LEHD and POMO](#for-fine-tuning-lehd-and-pomo)
- [Implementation](#implementation)
- [Basic Usage](#basic-usage)
  - [Attention bias via LLM Design](#attention-bias-via-llm-design)
  - [Fine-tuning Models](#fine-tuning-models)
  - [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [License](#license)
- [Citation](#citation)

## Dependencies

### For LLM Design
```bash
annotated-types==0.6.0
antlr4-python3-runtime==4.9.3
anyio==4.2.0
certifi==2024.7.4
distro==1.9.0
h11==0.14.0
httpcore==1.0.2
httpx==0.26.0
hydra-core==1.3.2
idna==3.7
numpy==1.23.3
omegaconf==2.3.0
openai==1.8.0
packaging==23.2
pydantic==2.5.3
pydantic_core==2.14.6
PyYAML==6.0.1
scipy==1.11.4
sniffio==1.3.0
tqdm==4.64.1
typing_extensions==4.9.0
```

### For LEHD and POMO
```bash
Python=3.8.6
torch==1.12.1
numpy==1.23.3
matplotlib==3.5.2
tqdm==4.64.1
pytz==2022.1
vrplib==1.0.0
```

If any package is missing, just install it following the prompts.

## Implementation

This project's structure is clear, the codes are based on .py files, and they should be easy to read, understand, and run.

## Basic Usage

### Attention bias via LLM Design

To run the code, execute *main.py*:
```bash
python main.py
```

### Fine-tuning Models

To fine-tune pre-trained models, i.e., LEHD-LLM and POMO-LLM, please run *train_ex.py* in each sub-folders TSP and CVRP:
```bash
# For TSP
python LEHD-LLM/TSP/train_ex.py
python POMO-LLM/NEW_py_ver/TSP/train_ex.py

# For CVRP
python LEHD-LLM/CVRP/train_ex.py
python POMO-LLM/NEW_py_ver/CVRP/train_ex.py
```


### Evaluation

To evaluate LEHD-LLM and POMO-LLM on synthetic datasets, run *test_ex.py* in each sub-folders TSP and CVRP:
```bash
# For TSP
python LEHD-LLM/TSP/test_ex.py
python POMO-LLM/NEW_py_ver/TSP/test_ex.py

# For CVRP
python LEHD-LLM/CVRP/test_ex.py
python POMO-LLM/NEW_py_ver/CVRP/test_ex.py
```

To evaluate LEHD-LLM and POMO-LLM on TSPLib and CVRPLibe, run *test_tsplib.py* and *test_vrplib.py* in each sub-folders TSP and CVRP:
```bash
# For TSP
python LEHD-LLM/TSP/test_tsplib.py
python POMO-LLM/NEW_py_ver/TSP/test_tsplib.py

# For CVRP
python LEHD-LLM/CVRP/test_vrplib.py
python POMO-LLM/NEW_py_ver/CVRP/test_vrplib.py
```

## Project Structure

```
checkpoints/
    lehd-lm_cvrp_checkpoint-73.pt
    lehd-lm_tsp_checkpoint-241.pt
    pomo-lm_cvrp_checkpoint-30700.pt
    pomo-lm_tsp_checkpoint-3300.pt
    pomo-lm_tsp_checkpoint-800.pt
LEHD-LLM/
    CVRP/
        .DS_Store
        heuristics.py
        test_ex.py
        test_vrplib.py
        Tester_inCVRPlib.py
        train_ex.py
        VRPEnv_ex.py
        VRPEnv_inCVRPlib.py
        ...
    TSP/
    utils/
LICENSE.md
LLM_design/
    cfg/
    main.py
    model/
    problems/
    prompts/
    requirements.txt
    utils/
POMO-LLM/
    NEW_py_ver/
README.md
```

## License
This project is licensed under the MIT License - see the *LICENSE.md* file for details.

## Citation
If you find this project useful, please cite our paper:
```
@inproceedings{
anonymous2025large,
title={Large Language Models powered Neural Solvers for Generalized Vehicle Routing Problems},
author={Anonymous},
booktitle={Towards Agentic AI for Science: Hypothesis Generation, Comprehension, Quantification, and Validation},
year={2025},
url={https://openreview.net/forum?id=EVqlVjvlt8}
}
```
