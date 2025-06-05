import pandas as pd
import yaml
import os
import re
from pathlib import Path
import sys
import logging

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from src.utils.text_cleaning import clean_text

# Configure logging
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'preprocess.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path)
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def remove_duplicates(df):
    return df.drop_duplicates()


def filter_leakage(df, keywords):
    pattern = "|".join([re.escape(keyword) for keyword in keywords])
    return df[~df["text"].str.contains(pattern, na=False)]


def normalize_text_length(df, min_tokens, max_tokens):
    return df[df["text"].str.split().str.len().between(min_tokens, max_tokens)]

def main():
    config_path = Path(os.environ["PREPROCESS_CONFIG"])
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    input_path = Path(config["input_path"])
    output_path = Path(config["output_path"])
    steps = config.get("preprocessing", {})

    df = pd.read_csv(input_path)
    original_count = len(df)
    logger.info(f"Loaded {original_count} rows from {input_path}")

    if steps.get("remove_duplicates", False):
        before = len(df)
        df = remove_duplicates(df)
        after = len(df)
        logger.info(f"Removed {before - after} duplicate rows")

    if steps.get("filter_leakage", {}).get("enabled", False):
        before = len(df)
        leakage_keywords = steps["filter_leakage"].get("leakage_keywords", [])
        df = filter_leakage(df, leakage_keywords)
        after = len(df)
        logger.info(f"Filtered {before - after} rows containing leakage keywords")

    if steps.get("normalize_text_length", {}).get("enabled", False):
        before = len(df)
        min_tokens = steps["normalize_text_length"].get("min_tokens", 0)
        max_tokens = steps["normalize_text_length"].get("max_tokens")
        df = normalize_text_length(df, min_tokens, max_tokens)
        after = len(df)
        logger.info(f"Normalized text length: removed {before - after} rows outside {min_tokens}-{max_tokens} tokens")

    if steps.get("clean_text", False):
        sample_before = df["text"].copy()
        df["text"] = df["text"].astype(str).apply(clean_text)
        cleaned = (sample_before != df["text"]).sum()
        logger.info(f"Cleaned text in {cleaned} rows using text_cleaning.clean_text()")

    df = df[["text", "author"]]
    logger.info(f"Final dataset has {len(df)} rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Wrote cleaned dataset to {output_path}")


if __name__ == "__main__":
    main()
