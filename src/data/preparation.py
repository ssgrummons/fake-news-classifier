import pandas as pd
from datasets import Dataset, ClassLabel
from src.features.tokenization import tokenizer, label2id

def preprocess_function(example: dict) -> dict:
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)

def prepare_dataset_from_csv(csv_path: str, label_col: str = "is_bs", num_labels: int = 2, label_names: list = None) -> Dataset:
    df = pd.read_csv(csv_path)
    df = df[["text", label_col]]
    df["labels"] = df[label_col]
    
    # Default label names: ["0", "1"]
    if label_names is None:
        label_names = [str(i) for i in range(num_labels)]
    
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset = hf_dataset.cast_column("labels", ClassLabel(num_classes=num_labels, names=label_names))
    
    return hf_dataset.map(preprocess_function, batched=True)

__all__ = ["prepare_dataset_from_csv"]
