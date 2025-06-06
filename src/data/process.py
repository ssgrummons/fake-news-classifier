import pandas as pd
import yaml
import os
import re
from pathlib import Path
import sys
import logging
import random
import hashlib

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

def deterministic_seed(text):
    # Create a deterministic integer seed from the text using SHA256
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % (10**8)

def mask_entities_per_row(text, entities):
    # Set a deterministic seed per row
    random.seed(deterministic_seed(text))

    mask_map = {}
    used_ids = set()
    
    # Replace longer entities first to avoid substring issues
    entities = sorted(entities, key=lambda x: -len(x[0]))
    for ent_text, ent_group in entities:
        if ent_text not in mask_map:
            # Generate a new random unused mask ID
            while True:
                mask_id = random.randint(1, 99999)
                if mask_id not in used_ids:
                    used_ids.add(mask_id)
                    break
            mask_token = f"<MASK_{mask_id}>"
            mask_map[ent_text] = mask_token

    # Replace all entity texts with their corresponding mask in the text
    for ent_text, mask_token in mask_map.items():
        pattern = re.compile(re.escape(ent_text))
        text = pattern.sub(mask_token, text)
    return text

def remove_duplicates(df):
    return df.drop_duplicates()

def normalize_text_length(df, min_tokens, max_tokens):
    return df[df["text"].str.split().str.len().between(min_tokens, max_tokens)]

def main():
    config_path = Path(os.environ["PROCESS_CONFIG"])
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    input_path      = Path(config["input_path"])       # e.g. data/interim/base.parquet
    output_path     = Path(config["train_output_path"])      # e.g. data/processed/train_dataset_1.csv
    output_path     = Path(config["test_output_path"])      # e.g. data/processed/test_dataset_1.csv
    steps           = config.get("processing", {})

    df = pd.read_parquet(input_path)
    original_count = len(df)
    logger.info(f"Loaded {original_count} rows from {input_path}")

    if steps.get("remove_duplicates", False):
        before = len(df)
        df = remove_duplicates(df)
        after = len(df)
        logger.info(f"Removed {before - after} duplicate rows")

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
        
    if steps.get("mask_entities", False):
        sample_before = df[["text", "entities"]].copy()
        df["text"] = df.apply(lambda row: mask_entities_per_row(row['text'], row['entities']), axis=1)
        cleaned = (sample_before != df["text"]).sum()
        logger.info(f"Masked entities in {cleaned} rows using mask_entities_per_row()")
        
        

    logger.info(f"Final dataset has {len(df)} rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Wrote cleaned dataset to {output_path}")


if __name__ == "__main__":
    main()
