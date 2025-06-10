import pandas as pd
from datasets import Dataset, ClassLabel
from src.features.tokenization import tokenizer

def preprocess_function(example: dict) -> dict:
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)

def prepare_dataset_from_csv(csv_path: str, label_col: str = "is_bs", num_labels: int = 2, label_names: list = None, use_ner: bool = False, ner_format: str = "append") -> Dataset:
    df = pd.read_csv(csv_path)
    df["labels"] = df[label_col]

    if label_names is None:
        label_names = [str(i) for i in range(num_labels)]

    # NER Feature Augmentation
    if use_ner and "entities" in df.columns:
        def merge_text(row):
            if ner_format == "append":
                return f"{row['text']} [SEP] {row['entities']}"
            elif ner_format == "prepend":
                return f"{row['entities']} [SEP] {row['text']}"
            return row["text"]

        df["text"] = df.apply(merge_text, axis=1)

    hf_dataset = Dataset.from_pandas(df)
    hf_dataset = hf_dataset.cast_column("labels", ClassLabel(num_classes=num_labels, names=label_names))
    return hf_dataset.map(preprocess_function, batched=True)


__all__ = ["prepare_dataset_from_csv"]
