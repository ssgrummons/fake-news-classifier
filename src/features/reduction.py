import numpy as np
import umap

def dimension_reduction(embeddings: np.ndarray, 
                        n_components: int = 2, 
                        random_state: int = 42) -> np.ndarray:
    """
    Reduces an array of vectors to n dimensions using UMAP

    Args:
        embeddings (np.ndarray): Multi-dimensional vector array
        n_components (int): Number of dimensions to reduce (defaults to 2)
        random_state (int): Random seed for UMAP initialization (defaults to 42)

    Returns:
        np.ndarray: 2 Dimensional Vector Array
    """
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    embeddings_red = reducer.fit_transform(embeddings)
    return embeddings_red