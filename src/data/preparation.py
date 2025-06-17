import pandas as pd
from datasets import Dataset, ClassLabel
from src.features.tokenization import tokenizer
import chardet
import logging
import os
from pathlib import Path
import sys

# Configure logging
project_root = Path(__file__).parent.parent.parent
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'preparation.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path)
    ]
)
logger = logging.getLogger(__name__)

def preprocess_function(example: dict) -> dict:
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)

def prepare_dataset_from_csv(csv_path: str, label_col: str = "is_bs", num_labels: int = 2, label_names: list = None, use_ner: bool = False, ner_format: str = "append") -> Dataset:
    with open(csv_path, 'rb') as f:
        result = chardet.detect(f.read())

    df = pd.read_csv(csv_path, encoding=result['encoding'])
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

    # Drop bad rows before tokenization
    initial_len = len(df)
    df = df[df["text"].apply(lambda x: isinstance(x, str) and x.strip() != "")]
    filtered_len = len(df)
    dropped = initial_len - filtered_len
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows with empty or invalid 'text' values.")

    hf_dataset = Dataset.from_pandas(df)
    hf_dataset = hf_dataset.cast_column("labels", ClassLabel(num_classes=num_labels, names=label_names))
    return hf_dataset.map(preprocess_function, batched=True)


def prepare_eval_dataset_from_csv(csv_path: str, config: dict) -> Dataset:
    data_config = config.get("data", {})
    model_config = config.get("model", {})

    return prepare_dataset_from_csv(
        csv_path=csv_path,
        label_col=data_config.get("label_column", "is_bs"),
        num_labels=model_config.get("num_labels", 2),
        label_names=None,
        use_ner=data_config.get("use_ner", False),
        ner_format=data_config.get("ner_format", "append")
    )


__all__ = ["prepare_dataset_from_csv"]
