import pandas as pd
import yaml
import os
import re
from pathlib import Path
import sys
import logging
import kagglehub
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import random
import hashlib
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner = pipeline("ner", model=model, tokenizer=tokenizer, device=0 if device == "mps" else -1, grouped_entities=True)


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

ner = None
vectorizer_model = None
topic_model = None

def load_vectorizer_model():
    global vectorizer_model
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
    return vectorizer_model

def load_topic_model(vectorizer_model):
    global topic_model
    topic_model = BERTopic(vectorizer_model=vectorizer_model)
    return topic_model

def load_ner_pipeline(model_name: str):
    global ner
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ner = pipeline("ner", model=model, tokenizer=tokenizer, device=0 if device == "mps" else -1, grouped_entities=True)

def download_data_from_kagglehub(dataset_name: str = "clmentbisaillon/fake-and-real-news-dataset", 
                                 version: str = "latest") -> str :
    path = kagglehub.download_dataset(dataset_name, version=version)
    return path

def concat_and_label_news_df(path: str,
                             label_name: str = 'is_bs') -> pd.DataFrame:
    # Load the news data
    df_fake = pd.read_csv(os.path.join(path, "fake.csv"))
    df_true = pd.read_csv(os.path.join(path, "true.csv"))
    # Concatenate the dataframes
    combined_df = pd.concat([df_fake, df_true], ignore_index=True)
    # Label the data
    combined_df[label_name] = combined_df.apply(lambda row: 1 if row.name in df_fake.index else 0, axis=1)
    return combined_df
    
def clean_news_data(df: pd.DataFrame, 
                    text_col: str = 'text',
                    label_col: str = 'is_bs') -> pd.DataFrame:
    # Filter columns
    df = df[[text_col, label_col]]
    # Remove the legitimate news articles that are not clearly labeled from Reuters. 
    # These are articles that just quote tweets. 
    unclean_negs = df[(~df[text_col].str.contains('\(Reuters\) -', case=True) & (df[label_col] == 0))]
    mask = ~df.index.isin(unclean_negs.index)
    df = df[mask]
    # Remove indications that the articles is from a News Source
    df[text_col] = df[text_col].str.replace(r'^.*\(Reuters\)\s*-\s*', '', regex=True)
    return df
    
def extract_entities(text: str) -> list:
    global ner
    if ner is None:
        raise RuntimeError("NER pipeline has not been initialized.")
    doc = ner(text)
    return [(ent['word'], ent['entity_group']) for ent in doc]

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

def apply_topic_modeling(texts: list) -> list:
    global topic_model, vectorizer_model
    if topic_model is None:
        vectorizer = load_vectorizer_model()
        topic_model = load_topic_model(vectorizer)
    topics, _ = topic_model.fit_transform(texts)
    return topics


def main():
    config_path = Path(os.environ["PREPROCESS_CONFIG"])
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    input_path = Path(config["input_path"])
    output_path = Path(config["output_path"])
    steps = config.get("preprocessing", {})
    
    if not input_path.exists():
        logger.info(f"{input_path} does not exist. Downloading raw dataset from KaggleHub.")
        raw_data_dir = download_data_from_kagglehub()
        df = concat_and_label_news_df(raw_data_dir, label_name="is_bs")
        df = clean_news_data(df, text_col="text", label_col="is_bs")
        input_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(input_path, index=False)
        logger.info(f"Saved initial cleaned and labeled data to {input_path}")

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
        
    if steps.get("mask_entities", {}).get("enabled", False):
        model_name = steps["mask_entities"].get("model_name", "dbmdz/bert-large-cased-finetuned-conll03-english")
        load_ner_pipeline(model_name)
        df["entities"] = df["text"].apply(extract_entities)
        logger.info(f"Extracted entities for {len(df)} rows using NER")

        
    if steps.get("apply_topic_model", False):
        df["topic"] = apply_topic_modeling(df["text"].tolist())
        logger.info(f"Assigned topic labels to {len(df)} rows")

    logger.info(f"Final dataset has {len(df)} rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Wrote cleaned dataset to {output_path}")


if __name__ == "__main__":
    main()
