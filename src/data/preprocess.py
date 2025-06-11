import pandas as pd
import yaml
import os
from pathlib import Path
import sys
import logging
import kagglehub
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from newspaper import Article
import feedparser

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

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

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

def download_data_from_kagglehub(dataset_name: str = "clmentbisaillon/fake-and-real-news-dataset") -> str :
    path = kagglehub.dataset_download(dataset_name)
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


def apply_topic_modeling(texts: list) -> list:
    global topic_model, vectorizer_model
    if topic_model is None:
        vectorizer = load_vectorizer_model()
        topic_model = load_topic_model(vectorizer)
    topics, _ = topic_model.fit_transform(texts)
    return topics

def scrape_rss_feed(feed_url, max_articles=1000):
    feed = feedparser.parse(feed_url)
    data = []
    for entry in feed.entries[:max_articles]:
        url, date = entry.link, entry.published
        art = Article(url)
        art.download(); art.parse()
        if len(art.text) > 200:  # filter for meatier content
            data.append({
                "url": url,
                "title": art.title,
                "date": date,
                "text": art.text[:1000]  # limit excerpt to safe use
            })
    return pd.DataFrame(data)

def create_df_from_feeds(feeds: list, max_articles=1000) -> pd.DataFrame:
    df_list = []
    for feed_url in feeds:
        df = scrape_rss_feed(feed_url, max_articles)
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

def label_and_clean_feed_df(legit_df: pd.DataFrame,
                            bs_df: pd.DataFrame,
                            text_col: str = 'text',
                            label_col: str = 'is_bs') -> pd.DataFrame:
    df = pd.concat([legit_df, bs_df], ignore_index=True)
    df[label_col] = df.apply(lambda row: 1 if row.name in bs_df.index else 0, axis=1)
    df = df[[text_col, label_col]]
    return df

def main():
    config_path = Path(os.environ["PREPROCESS_CONFIG"])
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    #output_path     = Path(config["output_path"]) 
    interim_path        = Path(config["interim_path"])
    dataset_name        = config["dataset_name"]
    entity_model_name   = config["entity_model_name"]
    label_name          = config["label_name"]
    feature_name        = config["feature_name"]
    legit_rss_feeds     = config["legit_rss_feeds"]
    bs_rss_feeds        = config["bs_rss_feeds"]
    max_articles        = config["max_articles"]
    
    logger.debug(f"Ensuring directory {interim_path} ...")
    interim_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading raw dataset from KaggleHub...")
    raw_data_dir = download_data_from_kagglehub(dataset_name)
    logger.info(f"Dataset downloaded to {raw_data_dir}")
    logger.info(f"Concatenating and Labeling data...")
    df = concat_and_label_news_df(raw_data_dir, label_name=label_name)
    original_count = len(df)
    logger.info(f"Initial Cleansing of {original_count} rows of data...")
    df = clean_news_data(df, text_col=feature_name, label_col=label_name)
    
    logger.info(f"Processing Legitimate RSS Feeds...")
    legit_df = create_df_from_feeds(legit_rss_feeds, max_articles)
    logger.info(f"Downloaded {legit_df.shape[0]} rows of legitimate RSS data...")
    logger.info(f"Processing BS RSS Feeds...")
    bs_df = create_df_from_feeds(bs_rss_feeds, max_articles)
    logger.info(f"Downloaded {bs_df.shape[0]} rows of BS RSS data...")
    logger.info("Concatenating and Labeling RSS Data...")
    rss_df = label_and_clean_feed_df(legit_df, bs_df, feature_name, label_name)
    logger.info("Merging RSS Data with News Data...")
    df = pd.concat([df, rss_df], ignore_index=True)
    logger.info(f"Processed {df.shape[0]} total rows of data")
    
    logger.info(f"Running NER pipeline with model {entity_model_name}...")
    load_ner_pipeline(entity_model_name)
    df["entities"] = df[feature_name].apply(extract_entities)
    logger.info(f"Extracted entities on {df.shape[0]} rows of data...")
    
    interim_topics_path = interim_path / "topics.parquet"
    logger.info("Running topic modeling...")
    load_vectorizer_model()
    load_topic_model(vectorizer_model)
    topics, probs = topic_model.fit_transform(df[feature_name].tolist())
    df["topic"] = topics
    
    logger.info(f"Extracted topics on {df.shape[0]} rows of data...")
    topic_counts = df.groupby(["topic", label_name]).size().unstack(fill_value=0)
    topic_counts["total"] = topic_counts.sum(axis=1)
    topic_counts["fake_ratio"] = topic_counts[1] / topic_counts["total"]
    topic_counts.columns.name = None
    topic_counts.reset_index(inplace=True)
    topic_counts.to_parquet(interim_topics_path, index=False)
    
    # df[[feature_name, "topic"]].to_parquet(interim_topics_path, index=False)
    logger.info(f"Wrote topic summary statistics to to {interim_topics_path}")

    logger.info(f"Final dataset has {len(df)} rows")
    output_path = interim_path / "base.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Wrote cleaned dataset to {output_path}")


if __name__ == "__main__":
    main()
