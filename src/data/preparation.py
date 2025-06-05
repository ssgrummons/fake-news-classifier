import pandas as pd
from datasets import Dataset, ClassLabel
from src.features.tokenization import tokenizer, label2id

def preprocess_function(example: dict) -> dict:
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)

def prepare_dataset_from_csv(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path)
    df = df[["text", "author"]]
    df['labels'] = df['author'].map(label2id)
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset = hf_dataset.cast_column("labels", ClassLabel(num_classes=4, names=list(label2id.keys())))
    return hf_dataset.map(preprocess_function, batched=True)

__all__ = ["prepare_dataset_from_csv"]
