import mlflow
import mlflow.transformers
from transformers import TrainerCallback, pipeline
from pathlib import Path
import shutil
import os
import json
import hashlib


def start_run(run_name: str = "transformer_experiment"):
    """Starts or resumes an MLflow run."""
    mlflow.set_experiment("transformer-text-classification")
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict):
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics(metrics: dict):
    for key, value in metrics.items():
        mlflow.log_metric(key, value)


def log_artifacts_from_dir(dir_path: str, artifact_path: str = "outputs"):
    """Logs all files in a directory as artifacts."""
    path_obj = Path(dir_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Artifact directory {dir_path} not found")
    mlflow.log_artifacts(dir_path, artifact_path)


def log_dataset_info(dataset_path: str, project_root: str = None):
    """Log dataset information including DVC tracking details."""
    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)
    
    dataset_path = Path(dataset_path)
    
    # Log basic dataset info as parameters (for backward compatibility)
    if dataset_path.exists():
        file_size = dataset_path.stat().st_size
        mlflow.log_param("dataset_path", str(dataset_path))
        mlflow.log_param("dataset_size_bytes", file_size)
        
        # Calculate file hash for data versioning
        with open(dataset_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        mlflow.log_param("dataset_md5_hash", file_hash)
        
        # Log dataset using MLflow's dataset logging feature
        try:
            from mlflow.data.pandas_dataset import PandasDataset
            import pandas as pd
            
            # Read the dataset to create MLflow dataset object
            df = pd.read_csv(dataset_path)
            dataset = mlflow.data.from_pandas(
                df, 
                source=str(dataset_path),
                name=f"training_dataset_{file_hash[:8]}",
                digest=file_hash
            )
            
            # Log the dataset input
            mlflow.log_input(dataset, context="training")
            
        except Exception as e:
            # Fallback: log as file-based dataset if pandas approach fails
            try:
                dataset = mlflow.data.from_csv(
                    str(dataset_path),
                    name=f"training_dataset_{file_hash[:8]}",
                    digest=file_hash
                )
                mlflow.log_input(dataset, context="training")
            except Exception as e2:
                mlflow.log_param("dataset_logging_error", str(e2))
    
    # Log DVC information if available
    dvc_file = dataset_path.with_suffix(dataset_path.suffix + '.dvc')
    if dvc_file.exists():
        mlflow.log_artifact(str(dvc_file), artifact_path="dvc")
        mlflow.log_param("dvc_tracked", True)
    else:
        mlflow.log_param("dvc_tracked", False)
    
    # Log dvc.lock if it exists in project root
    dvc_lock = project_root / "dvc.lock"
    if dvc_lock.exists():
        mlflow.log_artifact(str(dvc_lock), artifact_path="dvc")
    
    # Log dvc.yaml if it exists in project root  
    dvc_yaml = project_root / "dvc.yaml"
    if dvc_yaml.exists():
        mlflow.log_artifact(str(dvc_yaml), artifact_path="dvc")


def log_model_from_checkpoint(model, tokenizer, model_name: str = "transformer_model"):
    """Log HuggingFace model object and tokenizer as an MLflow model."""
    pipe = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None
    )

    # Use a raw string as input_example, not a dict
    input_example = "Example input text"

    mlflow.transformers.log_model(
        transformers_model=pipe,
        artifact_path=model_name,
        input_example=input_example
    )


def save_training_args(args, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args_path = output_dir / "training_args.json"
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    mlflow.log_artifact(str(args_path), artifact_path="config")


class MLflowCallback(TrainerCallback):
    """Custom callback to log evaluation metrics at the end of each epoch."""
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            for k, v in metrics.items():
                if isinstance(v, (float, int)):
                    mlflow.log_metric(k, v, step=state.global_step)

    def on_save(self, args, state, control, **kwargs):
        log_model_from_checkpoint(model=kwargs["model"], tokenizer=kwargs["tokenizer"])
        return control

    def on_train_end(self, args, state, control, **kwargs):
        save_training_args(args, args.output_dir)
