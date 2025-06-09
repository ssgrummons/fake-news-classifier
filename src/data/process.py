import pandas as pd
import yaml
import os
import re
from pathlib import Path
import sys
import logging
import random
import hashlib
from sklearn.model_selection import train_test_split
from collections import Counter

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from src.utils.text_cleaning import clean_text
from src.eda.richness import compute_lexical_metrics

# Configure logging
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'process.log')

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
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % (10**8)

def mask_entities_per_row(text, entities):
    random.seed(deterministic_seed(text))
    mask_map = {}
    used_ids = set()
    entities = sorted(entities, key=lambda x: -len(x[0]))
    for ent_text, ent_group in entities:
        if ent_text not in mask_map:
            while True:
                mask_id = random.randint(1, 99999)
                if mask_id not in used_ids:
                    used_ids.add(mask_id)
                    break
            mask_token = f"<MASK_{mask_id}>"
            mask_map[ent_text] = mask_token
    for ent_text, mask_token in mask_map.items():
        pattern = re.compile(re.escape(ent_text))
        text = pattern.sub(mask_token, text)
    return text

def remove_duplicates(df):
    return df.drop_duplicates(subset=["text"])

def normalize_text_length(df, min_tokens, max_tokens):
    return df[df["text"].str.split().str.len().between(min_tokens, max_tokens)]

def filter_low_lex_diversity(df: pd.DataFrame, min_hhd_threshold: float) -> pd.DataFrame:
    """
    Filters out rows with lexical diversity below the specified HDD threshold.
    
    Args:
        df (pd.DataFrame): Input DataFrame with a 'text' column.
        min_hhd_threshold (float): Minimum acceptable HDD value.

    Returns:
        pd.DataFrame: Filtered DataFrame with only rows meeting the lexical diversity requirement.
    """
    # Compute lexical diversity metrics
    df = compute_lexical_metrics(df, text_col="text")

    # Filter by HDD
    filtered_df = df[df["hdd"] >= min_hhd_threshold].copy()

    # Drop lexical metrics columns to keep output clean
    filtered_df.drop(columns=["ttr", "mtld", "hdd", "terms"], errors="ignore", inplace=True)

    return filtered_df

def run_data_processing(config: dict, df: pd.DataFrame) -> pd.DataFrame:
    steps = config.get("processing", {})
    
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
        
    if steps.get("mask_entities", {}).get("enabled", False):
        sample_before = df["text"].copy()
        df["text"] = df.apply(lambda row: mask_entities_per_row(row['text'], row['entities']), axis=1)
        cleaned = (sample_before != df["text"]).sum()
        logger.info(f"Masked entities in {cleaned} rows using mask_entities_per_row()")

    if steps.get("filter_low_lexical_diversity", {}).get("enabled", False):
        before = len(df)
        min_hhd_threshold = steps.get("filter_low_lexical_diversity", {}).get("min_hhd_threshold", 0.7)
        df = filter_low_lex_diversity(df, min_hhd_threshold)
        after = len(df)
        logger.info(f"Removed {before - after} rows with low lexical diversity using filter_low_lex_diversity()")

    return df

def run_data_splitting(config: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_cfg = config.get("splitting", {})
    
    train_ratio = split_cfg.get("train_ratio", 0.9)
    test_ratio = split_cfg.get("test_ratio", 0.1)
    assert abs(train_ratio + test_ratio - 1.0) < 1e-6, "Train/test ratios must sum to 1.0"

    # Step 1: Topic Parity Filtering (remove small topics)
    topic_parity_cfg = split_cfg.get("sample_by_topic_parity", {})
    if topic_parity_cfg.get("enabled", False):
        min_topic_size = topic_parity_cfg.get("min_topic_size", 100)
        topic_counts = df["topic"].value_counts()
        valid_topics = topic_counts[topic_counts >= min_topic_size].index
        before = len(df)
        df = df[df["topic"].isin(valid_topics)]
        after = len(df)
        logger.info(f"Filtered topics for parity: removed {before - after} rows from underrepresented topics")

    # Step 2: Stratified Sampling
    stratified_cfg = split_cfg.get("stratified_sample", {})
    if stratified_cfg.get("enabled", False):
        stratify_cols = stratified_cfg.get("stratify_by", [])
        sample_n = stratified_cfg.get("sample_n")

        if not stratify_cols or sample_n is None:
            raise ValueError("If stratified_sample is enabled, 'stratify_by' and 'sample_n' must be defined")

        stratify_key = df[stratify_cols].astype(str).agg("__".join, axis=1)
        class_counts = stratify_key.value_counts()
        min_per_class = min(sample_n // len(class_counts), class_counts.min())

        sampled_df_list = []
        for key, group in df.groupby(stratify_key):
            sampled_group = group.sample(n=min(min_per_class, len(group)), random_state=42)
            sampled_df_list.append(sampled_group)

        df = pd.concat(sampled_df_list).sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"Stratified sampling on {stratify_cols} complete: reduced to {len(df)} rows")

    # Step 3: Train/test split
    stratify_cols = stratified_cfg.get("stratify_by", [])
    stratify_vals = None
    if stratify_cols:
        stratify_vals = df[stratify_cols].astype(str).agg("__".join, axis=1)

    train_df, test_df = train_test_split(
        df,
        train_size=train_ratio,
        test_size=test_ratio,
        stratify=stratify_vals,
        random_state=42
    )
    logger.info(f"Split into train ({len(train_df)} rows) and test ({len(test_df)} rows) using stratify={stratify_cols}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def main():
    config_path = Path(os.environ["PROCESS_CONFIG"])
    config = load_config(config_path)

    input_path = Path(config["input_path"])
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path}")

    # Run splitting first
    train_df, test_df = run_data_splitting(config, df)

    # Process training data
    train_df = run_data_processing(config, train_df)
    
    # Save processed training data
    train_output_path = Path(config["train_output_path"])
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_output_path, index=False)
    logger.info(f"Wrote {len(train_df)} rows from processed training data to {train_output_path}")

    # Save raw test data (not processed)
    test_output_path = Path(config["test_output_path"])
    test_output_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(test_output_path, index=False)
    logger.info(f"Wrote {len(test_df)} rows from unprocessed test data to {test_output_path}")

if __name__ == "__main__":
    main()
