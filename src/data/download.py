import pandas as pd
import os
import yaml
from pathlib import Path
import sys
import logging
import kagglehub
from newspaper import Article
import feedparser
from urllib.parse import urlparse, quote
import requests
from typing import List, Dict, Any

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
#from src.utils.helpers import load_config

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

def query_cc_index(domain: str, 
                   match_pattern: str = "*", 
                   limit: int = 50, 
                   index: str ='CC-MAIN-2025-21-index'):
    """
    Queries the Common Crawl Index for a given domain and pattern.

    Args:
        domain (str): Domain to search (e.g., 'www.naturalnews.com')
        match_pattern (str): Pattern to match after domain (e.g., '*article*')
        limit (int): Maximum number of URLs to return

    Returns:
        List[str]: List of matching URLs
    """
    index_url = (
        f"http://index.commoncrawl.org/{index}?"
        f"url={quote(domain + '/' + match_pattern)}&output=json"
    )
    response = requests.get(index_url, stream=True)
    response.raise_for_status()

    urls = []
    for line in response.iter_lines():
        if line:
            record = eval(line.decode('utf-8'))  # Use `json.loads()` if unsure of eval safety
            url = record.get("url")
            if url and len(urls) < limit:
                urls.append(url)

    return urls

def scrape_articles(urls: list[str], max_words: int = 500) -> pd.DataFrame:
    """
    Scrape full text from a list of article URLs using newspaper3k.

    Args:
        urls (list[str]): List of article URLs.
        max_words (int): Max number of words to retain from article text.

    Returns:
        pd.DataFrame: DataFrame with columns ['url', 'title', 'text'].
    """
    data = []
    for i, url in enumerate(urls):
        try:
            article = Article(url)
            article.download()
            article.parse()
            words = article.text.split()
            truncated_text = " ".join(words[:max_words])
            data.append({
                "url": url,
                "title": article.title,
                "text": truncated_text
            })
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            continue

    return pd.DataFrame(data)

def scrape_and_label_cc_domains(
    domain_entries: List[Dict[str, str]],
    label: int,
    samples_per_domain: int,
    label_col: str = "is_bs",
    max_words: int = 500
) -> pd.DataFrame:
    """
    Scrape articles from CC-index for each domain entry and assign a label.

    Args:
        domain_entries (List[Dict[str, str]]): List of dicts with 'domain', 'pattern', and 'index'
        label (int): 1 for BS, 0 for legit
        samples_per_domain (int): How many URLs to scrape per domain
        label_col (str): Name of label column
        max_words (int): Max words per article to retain

    Returns:
        pd.DataFrame: DataFrame with columns ['url', 'title', 'text', label_col, 'source']
    """
    all_rows = []

    for entry in domain_entries:
        domain = entry["domain"]
        pattern = entry.get("pattern", "*")
        index = entry.get("index", "CC-MAIN-2025-21-index")
        try:
            urls = query_cc_index(domain, match_pattern=pattern, limit=samples_per_domain, index=index)
            df = scrape_articles(urls, max_words=max_words)
            df[label_col] = label
            df["source"] = f"cc:{domain}"
            all_rows.append(df)
            logger.info(f"Scraped {len(df)} samples from {domain}")
        except Exception as e:
            logger.warning(f"Failed to scrape from {domain}: {e}")
            continue

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=["url", "title", "text", label_col, "source"])

def main():
    config_path = Path(os.environ["BASE_CONFIG"])
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    output_path         = Path(config["output_path"]) 
    dataset_name        = config["dataset_name"]
    label_name          = config["label_name"]
    feature_name        = config["feature_name"]
    legit_rss_feeds     = config["legit_rss_feeds"]
    bs_rss_feeds        = config["bs_rss_feeds"]
    max_articles        = config["max_articles"]
    bs_domains          = config["cc_index_scraping"].get("bs_domains", [])
    legit_domains       = config["cc_index_scraping"].get("legit_domains", [])
    n                   = config["cc_index_scraping"].get("samples_per_domain", 10)
    
    
    logger.debug(f"Ensuring directory {output_path} ...")
    output_path.mkdir(parents=True, exist_ok=True)
    
    ### Download Fake and Real News Raw Data from KaggleHub
    logger.info(f"Downloading raw dataset from KaggleHub...")
    raw_data_dir = download_data_from_kagglehub(dataset_name)
    logger.info(f"Dataset downloaded to {raw_data_dir}")
    logger.info(f"Concatenating and Labeling data...")
    df = concat_and_label_news_df(raw_data_dir, label_name=label_name)
    original_count = len(df)
    logger.info(f"Initial Cleansing of {original_count} rows of data...")
    df = clean_news_data(df, text_col=feature_name, label_col=label_name)
    logger.info(f"Processed {len(df)} rows of data from KaggleHub")
    df['source'] = dataset_name
    
    ### Scrape Data from CC-Index
    logger.info("Scraping BS domains from CC-index...")
    cc_bs_df = scrape_and_label_cc_domains(
        domain_entries=bs_domains,
        label=1,
        samples_per_domain=n,
        label_col=label_name
    )

    logger.info("Scraping legit domains from CC-index...")
    cc_legit_df = scrape_and_label_cc_domains(
        domain_entries=legit_domains,
        label=0,
        samples_per_domain=n,
        label_col=label_name
    )

    cc_df = pd.concat([cc_bs_df, cc_legit_df], ignore_index=True)
    logger.info(f"Collected {cc_df.shape[0]} rows from CC-index scraping...")
    logger.info("Appending CC-index data to master DataFrame...")
    cc_df = cc_df[[feature_name, label_name]]
    cc_df['source'] = 'CC-index'
    df = pd.concat([df, cc_df], ignore_index=True)
    
    ### Collect Data from RSS Feeds
    logger.info(f"Processing Legitimate RSS Feeds...")
    legit_df = create_df_from_feeds(legit_rss_feeds, max_articles)
    logger.info(f"Downloaded {legit_df.shape[0]} rows of legitimate RSS data...")
    logger.info(f"Processing BS RSS Feeds...")
    bs_df = create_df_from_feeds(bs_rss_feeds, max_articles)
    logger.info(f"Downloaded {bs_df.shape[0]} rows of BS RSS data...")
    logger.info("Concatenating and Labeling RSS Data...")
    rss_df = label_and_clean_feed_df(legit_df, bs_df, feature_name, label_name)
    rss_df['source'] = 'RSS'
    logger.info("Merging RSS Data with News Data...")
    df = pd.concat([df, rss_df], ignore_index=True)
    logger.info(f"Processed {df.shape[0]} total rows of data")

    logger.info(f"Final dataset has {len(df)} rows")
    output_path = output_path / "base.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Wrote cleaned dataset to {output_path}")

if __name__ == "__main__":
    main()