# Fake News Classification Model

This repository contains a modular and reproducible machine learning pipeline for transformer-based text classification. It supports multiple datasets, full ML experiment tracking with MLflow, dataset versioning with DVC, and a clean separation between preprocessing, tokenization, modeling, and evaluation.

## Project Overview

### Business Problem

The working assumption is that these different types of content will have different styles, vocabularies, topics, and tone that will allow us to infer the correct classification based on those semantic clues as opposed to content.

This project is designed to do an initial exploration of that hypothesis and then go through the process of training a transformer model to accurately classify unlabeled texts.  To that end, we need to set up this project with these capabilities:

* Enable rapid experimentation with multiple datasets
* Track experiment results and models 
* Reproduce data assets 
* Keep code cleanly separated across EDA, preprocessing, feature engineering, and training
* Automate functions for production pipeline deployments

### Key Features

* Modular Python package structure under `src/`
* Text classification model fine-tuning with [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/index)
* Support for multiple experiments via config files
* Tokenization decoupled from text cleansing
* Logging for metrics, parameters, and model artifacts with [MLflow](https://mlflow.org/)
* Managing data processing pipelines and versioning data assets with [DVC](https://dvc.org/)

---

## Project Structure

```bash
.
├── config/                 # YAML configuration files
│   ├── dvc/
│   ├── experiments/
│   ├── inference/
│   └── pipelines/
├── data/                   # Data files
│   ├── raw/
│   ├── processed/
│   └── inference/
├── notebooks/              # EDA and modeling exploration
├── src/                    # Modular source code
│   ├── data/               # Preprocessing and Hugging Face dataset prep
│   ├── eda/                # EDA helper functions
│   ├── features/           # Embeddings and tokenization
│   ├── inference/          # Inference script
│   ├── mlflow/             # Experiment Tracking functions
│   ├── models/             # Metrics and training script
│   ├── pipelines/          # Main experiment runner
│   └── utils/              # Miscellaneous helpers
├── logs/                   # Training logs
├── mlruns/                 # Experiment outputs
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── dvc.yaml / dvc.lock     # DVC pipeline definition and lock file
└── tests/                  # Unit tests (coming soon)
```

---

## Exploratory Data Analysis

All exploratory data analysis (EDA) was performed in Jupyter notebooks under the `notebooks/` directory. This includes:

* Class distribution analysis
* Richness metrics and lexical diversity
* TF-IDF baselines
* Text cleaning assessments
* Token length distribution visualizations

Notebooks are labeled in order and well documented.  Relevant concepts and findings validating the initial hypothesis are dicussed in detail there.

---

## Data Preprocessing & Feature Engineering

* `src/data/preprocess.py`: Handles normalization, lowercasing, and whitespace stripping.
* `src/data/preparation.py`: Loads cleansed CSVs and tokenizes them into Hugging Face `DatasetDict` objects.
* `src/features/tokenization.py`: Defines the tokenizer, label mappings, and tokenizer class.

This modularization allows flexible experimentation with different data sources and tokenization strategies.

---

## Model Training

Transformer-based model training is handled by:

* `src/models/train_transformer_model.py`: Generalized training pipeline using Hugging Face `Trainer` with early stopping and MLflow integration.
* Config-driven design: Training behavior is configured using YAML files located in `config/experiments/`.

Use `src/pipelines/run_experiments.py` to run one or more experiments as defined in your config.

---

## Experiment Tracking

All training runs:

* Log parameters, metrics, and models using MLflow
* Support early stopping and `metric_for_best_model` tracking
* Save the best checkpoint for later inference or comparison

Logs and metrics are saved under the `logs/` and `experiments/` directories.

---

## Setup & Installation

### Conda Environment

To replicate the environment:

```bash
conda create -n transformer-pipeline python=3.10
conda activate transformer-pipeline
```

### Non-Pip Dependencies

Install the following dependencies manually **before** using `pip`:

```bash
# Required for tokenizers and certain NLP backends
conda install -c conda-forge rust
conda install -c conda-forge libpython

# Optional but recommended
conda install -c conda-forge jupyterlab
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Install DVC

Use DVC's [installation guide](https://dvc.org/doc/install) to install DVC on your platform.

Initialize as follows:
```bash
dvc init
git add .dvc .gitignore
git commit -m "Initialize DVC"
```

---

## Running the Pipeline

### DVC Preprocessing Pipelines

Use the configs under `config/dvc/` to run dataset-specific preprocessing.  The following command is ran from the project root directory and uses the `dvc.yaml` file.

```bash
dvc repro
```

### Running Experiments

Specify a list of experiment configs in your pipeline config file then run:

```bash
python src/pipelines/run_experiments.py 
```

The script will iterate through all the experiment configuration files listed in `config/pipelines/experiments.yaml`.

---

## Inference

Run the following to run batch inference using the parameters specified in `config/inference/inference.yaml`:

```bash
python src/inference/run_batch_inference.py
```

---

## Tests

Test structure scaffolding is present in `tests/`, and unit tests will be added as the pipeline matures.

