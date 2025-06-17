import pandas as pd
import yaml
import os
from pathlib import Path
import sys
import logging
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from umap import UMAP
from typing import List, Dict, Any


project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from src.utils.helpers import load_config

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


def load_vectorizer_model(embedding_model_name: str = "all-MiniLM-L6-v2"):
    global vectorizer_model
    vectorizer_model = SentenceTransformer(embedding_model_name)
    return vectorizer_model

def load_topic_model(vectorizer_model, 
                     n_neighbors: int = 30, 
                     n_components: int = 5, 
                     min_dist: float = 0.4, 
                     metric: str = 'euclidean',
                     min_cluster_size: int = 60, 
                     cluster_selection_method: str = 'eom'):
    global topic_model

    # Note: UMAP often works better with 'cosine' metric for embeddings
    umap_model = UMAP(
        n_neighbors=n_neighbors, 
        n_components=n_components, 
        min_dist=min_dist, 
        metric='cosine'  # override here if you're using embedding distances
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size, 
        metric=metric, 
        cluster_selection_method=cluster_selection_method
    )

    topic_model = BERTopic(
        embedding_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        verbose=True
    )
    return topic_model

def load_ner_pipeline(model_name: str):
    global ner
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ner = pipeline("ner", model=model, tokenizer=tokenizer, device=0 if device == "mps" else -1, grouped_entities=True)
    
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

def main():
    config_path = Path(os.environ["PREPROCESS_CONFIG"])
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    interim_path                = Path(config["interim_path"])
    input_path                  = Path(config["input_path"])
    entity_model_name           = config["entity_model_name"]
    file_name                   = config["file_name"]
    embedding_model_name        = config["embedding_model_name"]
    n_neighbors                 = config["n_neighbors"]
    n_components                = config["n_components"]
    min_dist                    = config["min_dist"]
    metric                      = config["metric"]
    min_cluster_size            = config["min_cluster_size"]
    cluster_selection_method    = config["cluster_selection_method"]
    feature_name                = config["feature_name"]
    
    logger.debug(f"Ensuring directory {interim_path} ...")
    interim_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_parquet(input_path / file_name)
    
    ### Perform Entity Recognition
    logger.info(f"Running NER pipeline with model {entity_model_name}...")
    load_ner_pipeline(entity_model_name)
    df["entities"] = df[feature_name].apply(extract_entities)
    logger.info(f"Extracted entities on {df.shape[0]} rows of data...")
    
    ### Perform Topic Modeling
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
