import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.train_transformer_model import run_training_from_config_path
import yaml
import traceback

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_all_experiments():
    pipeline_config = load_yaml("config/pipelines/experiments.yaml")
    experiment_paths = pipeline_config.get("experiments", [])

    for config_path in experiment_paths:
        print(f"\n[INFO] Running experiment: {config_path}")
        try:
            run_training_from_config_path(config_path)
        except Exception:
            print(f"[ERROR] Failed experiment: {config_path}")
            print(traceback.format_exc())

if __name__ == "__main__":
    run_all_experiments()
