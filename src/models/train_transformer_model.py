#!/usr/bin/env python3
"""
Transformer model training script for NLP classification experiments.
This script is designed to be run from the project root directory.
"""

import argparse
import logging
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
from collections import Counter

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from transformers import (
    TrainingArguments, 
    Trainer, 
    AutoModelForSequenceClassification, 
    EarlyStoppingCallback
)
from dotenv import load_dotenv

# Import your custom modules
from src.data.preparation import prepare_dataset_from_csv
from src.features.tokenization import tokenizer
from src.models.metrics import compute_metrics
from src.mflow.experiment_tracking import start_run, log_params, log_metrics, log_model_from_checkpoint, log_dataset_info

# Load environment variables
load_dotenv()

# Configure logging
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'training.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path)
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Successfully loaded config from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
        raise


def setup_mlflow_tracking_uri(mlruns_dir: str):
    """Set up MLFlow tracking URI to use specified directory."""
    import mlflow
    tracking_uri = f"file://{os.path.abspath(mlruns_dir)}"
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLFlow tracking URI set to: {tracking_uri}")


def validate_dataset_split(train_dataset, eval_dataset):
    """Validate that all classes are represented in both train and eval sets."""
    train_counter = Counter(train_dataset["labels"])
    eval_counter = Counter(eval_dataset["labels"])
    
    logger.info("Train dataset label distribution:")
    for label, count in train_counter.items():
        logger.info(f"  Label {label}: {count} samples")
    
    logger.info("Eval dataset label distribution:")
    for label, count in eval_counter.items():
        logger.info(f"  Label {label}: {count} samples")
    
    # Check if all classes are represented in both sets
    train_labels = set(train_counter.keys())
    eval_labels = set(eval_counter.keys())
    
    if train_labels != eval_labels:
        logger.warning(f"Label mismatch between train and eval sets!")
        logger.warning(f"Train labels: {train_labels}")
        logger.warning(f"Eval labels: {eval_labels}")


def create_model(config: dict):
    """Create and configure the transformer model."""
    model_config = config['model']
    
    logger.info(f"Loading model: {model_config['model_name']}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config['model_name'],
        num_labels=model_config['num_labels']
    )
    
    if model_config.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    return model


def create_training_arguments(config: dict) -> TrainingArguments:
    """Create TrainingArguments from config."""
    training_config = config['training']
    
    # Handle output directory - make it relative to project root
    output_dir = training_config.get('output_dir', 'results')
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=float(training_config.get('learning_rate', 2e-5)),
        per_device_train_batch_size=int(training_config.get('per_device_train_batch_size', 4)),
        per_device_eval_batch_size=int(training_config.get('per_device_eval_batch_size', 4)),
        gradient_accumulation_steps=int(training_config.get('gradient_accumulation_steps', 4)),
        num_train_epochs=int(training_config.get('num_train_epochs', 2)),
        weight_decay=float(training_config.get('weight_decay', 0.01)),
        eval_strategy=training_config.get('eval_strategy', 'epoch'),
        save_strategy=training_config.get('save_strategy', 'epoch'),
        report_to=training_config.get('report_to', []),
        load_best_model_at_end=bool(training_config.get('load_best_model_at_end', True)),
        metric_for_best_model=training_config.get('metric_for_best_model', 'f1_macro'),
        greater_is_better=bool(training_config.get('greater_is_better', True)),
        push_to_hub=bool(training_config.get('push_to_hub', False)),
        fp16=bool(training_config.get('fp16', False)),
        dataloader_num_workers=int(training_config.get('dataloader_num_workers', 0)),
        dataloader_pin_memory=bool(training_config.get('dataloader_pin_memory', False))
    )
    
    logger.info("Training arguments created successfully")
    return training_args


def prepare_data(config: dict):
    """Prepare training and evaluation datasets."""
    data_config = config['data']
    
    # Handle data path - make it relative to project root if not absolute
    data_path = data_config['train_csv_path']
    if not os.path.isabs(data_path):
        data_path = os.path.join(project_root, data_path)
    
    logger.info(f"Loading dataset from: {data_path}")
    model_config = config['model']
    data_config = config['data']

    dataset = prepare_dataset_from_csv(
        csv_path=data_path,
        label_col=data_config.get('label_column', 'is_bs'),
        num_labels=model_config.get('num_labels', 2)
    )
    
    # Split dataset
    test_size = data_config.get('test_size', 0.2)
    stratify_column = data_config.get('stratify_by_column', 'labels')
    
    logger.info(f"Splitting dataset with test_size={test_size}, stratify_by={stratify_column}")
    split_dataset = dataset.train_test_split(
        test_size=test_size, 
        stratify_by_column=stratify_column
    )
    
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    logger.info(f"Dataset split completed - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # Validate the split
    validate_dataset_split(train_dataset, eval_dataset)
    
    return train_dataset, eval_dataset


def train_model(config: dict):
    """Main training function."""
    experiment_config = config['experiment']
    
    # Setup MLFlow tracking directory
    mlruns_dir = config.get('mlflow', {}).get('mlruns_dir', 'mlruns')
    if not os.path.isabs(mlruns_dir):
        mlruns_dir = os.path.join(project_root, mlruns_dir)
    setup_mlflow_tracking_uri(mlruns_dir)
    
    # Prepare data
    train_dataset, eval_dataset = prepare_data(config)
    
    # Create model
    model = create_model(config)
    
    # Create training arguments
    training_args = create_training_arguments(config)
    
    # Generate run name from experiment name + timestamp
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d_%H:%M")
    experiment_name = experiment_config.get('name', 'transformer_experiment')
    run_name = f"{experiment_name}_{formatted_date}"
    
    logger.info(f"Starting MLFlow run: {run_name}")
    
    # Start MLFlow run and train
    with start_run(run_name=run_name):
        # Create trainer
        early_stopping_patience = config.get('training', {}).get('early_stopping_patience', 2)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        )
        
        # Log parameters
        log_params(vars(training_args))
        
        # Log dataset information
        data_path = config['data']['train_csv_path']
        if not os.path.isabs(data_path):
            data_path = os.path.join(project_root, data_path)
        log_dataset_info(data_path, project_root)
        
        # Log additional config parameters
        log_params({
            'model_name': config['model']['model_name'],
            'early_stopping_patience': early_stopping_patience,
            'experiment_name': experiment_config.get('name', 'unnamed_experiment')
        })
        
        logger.info("Starting training...")
        trainer.train()
        
        logger.info("Training completed. Running evaluation...")
        eval_metrics = trainer.evaluate()
        log_metrics(eval_metrics)
        
        logger.info("Logging model...")
        log_model_from_checkpoint(model=trainer.model, tokenizer=tokenizer)
        
        logger.info("Final evaluation metrics:")
        for k, v in eval_metrics.items():
            logger.info(f"  {k}: {v}")
        
        logger.info("Training pipeline completed successfully!")

def run_training_from_config_path(config_path: str):
    """
    Run training from a YAML config path without using argparse.
    This is functionally equivalent to calling the script from CLI with --config.
    """
    config = load_config(config_path)
    
    logger.info(f"Starting training with config: {config_path}")
    logger.info(f"Experiment: {config.get('experiment', {}).get('name', 'unnamed')}")
    
    try:
        train_model(config)
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train transformer model with config file')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to YAML configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    logger.info(f"Starting training with config: {args.config}")
    logger.info(f"Experiment: {config.get('experiment', {}).get('name', 'unnamed')}")
    
    try:
        train_model(config)
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()