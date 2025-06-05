import yaml
import pandas as pd
import mlflow
from mlflow import pyfunc
from pathlib import Path

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_model_uri(model_name: str, alias: str = None):
    if alias:
        return f"models:/{model_name}@{alias}"
    else:
        return f"models:/{model_name}"

def load_model(model_uri: str):
    return mlflow.pyfunc.load_model(model_uri)

def run_inference(model, df, text_column: str):
    predictions = model.predict(df[text_column].tolist())

    if isinstance(predictions, pd.DataFrame):
        return predictions  # e.g., already has 'label', 'score'

    if isinstance(predictions, list):
        if isinstance(predictions[0], list):  # return_all_scores=True
            return pd.DataFrame([
                {
                    "label": max(p, key=lambda x: x["score"])["label"],
                    "score": max(p, key=lambda x: x["score"])["score"]
                } for p in predictions
            ])
        elif isinstance(predictions[0], dict):  # return_all_scores=False
            return pd.DataFrame([
                {"label": p["label"], "score": p.get("score", None)} for p in predictions
            ])
    raise ValueError("Unexpected prediction format")


def main():
    config = load_config("./config/inference/inference.yaml")

    model_uri = get_model_uri(
        model_name=config["model_name"],
        alias=config.get("alias")  # Optional key
    )

    model = load_model(model_uri)
    df = pd.read_csv(config["input_path"], index_col=0)
    predictions_df = run_inference(model, df, config["text_column"])

    # Minimal output: only include text and label
    minimal_df = df[[config["text_column"]]].copy()
    minimal_df[config["output_column"]] = predictions_df["label"]

    # Full output: include score for inspection
    full_df = df[[config["text_column"]]].copy()
    full_df[config["output_column"]] = predictions_df["label"]
    full_df["score"] = predictions_df["score"]

    # Write outputs
    output_path = Path(config["output_path"])
    scored_output_path = Path(config["scored_output_path"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    minimal_df.to_csv(output_path)
    full_df.to_csv(scored_output_path, index=False)

if __name__ == "__main__":
    main()
