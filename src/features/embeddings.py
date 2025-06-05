from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

def vectorize_text_array(text_array: pd.core.series.Series,
                         model_name: str) -> np.ndarray:
    """
    Vectorize an array of text using a pre-trained Sentence Transformer model.
    
    Args:
        text_array (pd.core.series.Series): Series of text data.
        model_name (str): Name of the pre-trained Sentence Transformer model.
    
    Returns:
        np.ndarray: Multi-dimensional Vector Array of embeddings.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_array)
    return embeddings