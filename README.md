# Empowering Large Scale Generalization of Neural Solvers through Large Language Models for Vehicle Routing Problems

**This code develop an efficient LLM-guied fine-tuning to enhance the large-scale generalization of Neural Solvers for solving TSP and CVRP.** 

## Dependencies

**For LLM design**
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

**For LEHD and POMO**
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

**For Attention Bias via LLM design**
To run the code, run *main.py*

**For Fine-tuning models**
To fine-tune pre-trained models, i.e., LEHD-LLM and POMO-LLM, please run *train_ex.py* in each sub-folders TSP and CVRP in each LEHD and POMO folder.

**For evaluation**

To evaluate LEHD-LLM and POMO-LLM on synthetic datasets, run *test_ex.py* in each sub-folders.

To evaluate LEHD-LLM and POMO-LLM on TSPLib and CVRPLibe, run *test_tsplib.py* and *test_vrplib.py* in each sub-folders.

