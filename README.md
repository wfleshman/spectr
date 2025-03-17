<div align="center">

<h1>SpectR: Dynamically Composing LM Experts
with Spectral Routing </h1>

</div>

This repo is a fork of the 2024 NeurIPS LLM-Merging competition starter code [here](https://github.com/llm-merging/LLM-Merging). 

## Setup Environment 

```
conda env create -f environment.yml --name llm-merging
conda activate llm-merging 
export PYTHONPATH=`pwd`
python llm_merging/setup.py install 
```

## Fetch Datasets
```
mkdir data
python make_datasets.py
```

## Fine-Tune Adapters

```
export HF_HOME=<path to your huggingface cache>
export HF_AUTH_TOKEN=<your huggingface auth token>
export OUT_DIR=<path to save adapters>

# choose model to fine-tune
export FT_MODEL="meta-llama/Llama-3.2-3B-Instruct"

# fit adapters on all datasets
python finetune.py
```

## Run Eval 

```
export HF_HOME=<path to your huggingface cache>
export HF_AUTH_TOKEN=<your huggingface auth token>

# choose model to eval
export EVAL_MODEL="meta-llama/Llama-3.2-3B-Instruct"
export MODEL_DIR=<path to your EVAL_MODEL adapters>

python llm_merging/main.py -m spectr
```