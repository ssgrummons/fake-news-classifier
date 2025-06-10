# Bullshit Classification Model

In Harry Frankfurt's book, **On Bullshit**,[^1] Frankfurt distinguishes between lies and bullshit.  The liar is concerned about the truth and ensuring the other party does not find the truth.  The bullshittter, however, is not interested in whether what they say is true or false; truth is irrelevant to their goal.  

I have often joked in the past about developing a "bullshotmeter," a tricorder-type device that would beep in the presence of bullshit.  This project is an attempt to bring that tech to life.  

The hypothesis is that bullshit has its own semantic patterns.  Maybe it's more bombastic, more sure of itself, less nuanced, more focused on eliciting some kind of response from the other party.  This project is an attempt to model those semantic patterns and use them to develop a classifier for bullshit.

## Project Overview

### Current Project Status

_This project is not complete yet!_

This project was built on the [Kaggle Fake News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data) dataset.  The dataset contains a mix of news articles and their corresponding labels (fake or real). 

The EDA phase of the project highlighted a number of the challenges to developing a model that can generalize outside this dataset.
- The article topics were imbalanced between real and fake news articles.  So it is likely that a model trained on this dataset will just learn to recognize topics most associated with bullshit and not the semantic patterns of bullshit.
- The vocabulary of fake and real news articles was very distinct, enabling a simple Vectorized TF-IDF model to perform well on the dataset.  That might be sufficient for a simple social media classifier, but it may not generalize from bullshit in the news to bullshit on LinkedIn or the bullshit stories from your brother-in-law. 
- The project right now is limited to the labeled data available.  To test if the model can generalize outside of this dataset, we will need to collect more tweets from influencers, more stories from your uncle, and more corporate memos about company culture.

### Project Structure

Even with those limitations of the project as is, this repository does demonstrate valuable features that can be leveraged in other projects.  Here are some of the key features:

* Modular Python package structure under `src/`
* Text classification model fine-tuning with [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/index)
* Support for multiple experiments via config files
* Tokenization decoupled from text cleansing
* Logging for metrics, parameters, and model artifacts with [MLflow](https://mlflow.org/)
* Managing data processing pipelines and versioning data assets with [DVC](https://dvc.org/)

These features enable the following:

* Rapid experimentation with multiple datasets
* Track experiment results and models 
* Reproduce data assets 
* Keep code cleanly separated across EDA, preprocessing, feature engineering, and training
* Automate functions for production pipeline deployments

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
│   ├── interim/
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

Notebooks are labeled in order and well documented.  

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

[^1]: Frankfurt, Harry G. On Bullshit. Princeton University Press, 2005.