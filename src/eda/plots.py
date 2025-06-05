import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D projection


def plot_boxplot(df: pd.DataFrame, metric: str, group_col: str = 'author', title_prefix: str = ''):
    """
    Plots a boxplot of the computed lengths grouped by a specified column.
    
    Args:
        df (pd.DataFrame): Dataframe with the computed length column.
        metric (str): Name of the computed length column.
        group_col (str, optional): Name of the column to group by. Defaults to 'author'.
        title_prefix (str, optional): Prefix for the plot title. Defaults to an empty string.
    
    Returns:
        None: Displays the boxplot.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=group_col, y=metric)
    plt.title(f"{title_prefix} Distribution by {group_col.capitalize()}")
    plt.xlabel(group_col.capitalize())
    plt.ylabel(f"{metric.replace('_', ' ').capitalize()}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_cluster_label_distribution(df: pd.DataFrame, 
                                    kmeans_model: KMeans):
    """
    Plot the distribution of cluster labels vs. author labels.  Used to identify clusters with high label overlap.
    
    Args:
        df (pd.DataFrame): DataFrame containing the text and author labels.
        kmeans_model (KMeans): KMeans model used to cluster the data. 
    
    Returns:
        None: Displays a bar chart showing the number of each cluster/label pair.
    """
    labels = kmeans_model.labels_
    df['cluster'] = labels

    # 3. Check the corresponding author labels with the cluster labels
    df['bs_cluster'] = df['cluster'].map(df.groupby('is_bs')['text'].transform('count'))

    # 4. Check the distribution of author counts in each cluster
    plt.figure(figsize=(10, 6))
    df.groupby('cluster')['is_bs'].value_counts().plot(kind='bar')
    plt.title('Distribution of Author Counts in Each Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()
    
    
def plot_embeddings_with_labels(embeddings: np.ndarray, 
                                labels: pd.core.series.Series,
                                label_name: str):
    """
    Plots a 2 dimensional vector array on a scatter plot, color coded by labels.
    Labels can be those from the dataset or from k-means clustering.
    
    Args:
        embeddings (np.ndarray): 2 dimensional vector array
        labels (Pandas Series): Labels for each vector
        label_name (str): Name of the label column in the labels DataFrame
        
    Returns:
        None: Displays a scatter plot with labels colored by their respective values.
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='viridis', s=10)
    plt.title(f"UMAP Plot Colored by {label_name}")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.colorbar(label=f'{label_name} Label')
    plt.grid(True)
    plt.show()
    
def plot_embeddings_3d_with_labels(embeddings: np.ndarray, 
                                   labels: pd.core.series.Series,
                                   label_name: str):
    """
    Plots a 3-dimensional vector array in a 3D scatter plot, color coded by labels.

    Args:
        embeddings (np.ndarray): 3D vector array of shape (n_samples, 3)
        labels (Pandas Series): Labels for each vector
        label_name (str): Name for the colorbar

    Returns:
        None: Displays a 3D scatter plot.
    """
    from mpl_toolkits.mplot3d import Axes3D  # Ensure 3D plotting works
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], 
                    c=labels, cmap='viridis', s=15)

    ax.set_title(f"3D UMAP Plot Colored by {label_name}")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")

    fig.colorbar(sc, ax=ax, label=f'{label_name} Label')
    plt.show()


def plot_3d_umap_plotly(embeddings: np.ndarray, 
                        labels: pd.Series, 
                        label_name: str = "Label"):
    """
    Plots a 3D UMAP scatter plot with interactive rotation using Plotly.

    Args:
        embeddings (np.ndarray): Array of shape (n_samples, 3)
        labels (pd.Series): Labels to color by
        label_name (str): Title for color legend

    Returns:
        None: Displays interactive plot.
    """
    df_plot = pd.DataFrame({
        "UMAP1": embeddings[:, 0],
        "UMAP2": embeddings[:, 1],
        "UMAP3": embeddings[:, 2],
        label_name: labels.values
    })

    fig = px.scatter_3d(df_plot, 
                        x="UMAP1", y="UMAP2", z="UMAP3", 
                        color=label_name, 
                        opacity=0.7,
                        size_max=10)

    fig.update_layout(
        title=f"3D UMAP Plot Colored by {label_name}",
        scene=dict(
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            zaxis_title="UMAP3"
        ),
        width=800,
        height=700
    )
    
    fig.show()
    

def plot_3d_umap_by_label_layers(embeddings: np.ndarray,
                                  labels: pd.Series,
                                  label_name: str = "Label"):
    """
    Plots a 3D UMAP scatter plot using Plotly, plotting label groups in layers
    to prevent dominant labels from obscuring others.

    Args:
        embeddings (np.ndarray): UMAP-reduced embeddings (n_samples, 3)
        labels (pd.Series): Categorical labels for color coding
        label_name (str): Label for legend

    Returns:
        None
    """
    df_plot = pd.DataFrame({
        "UMAP1": embeddings[:, 0],
        "UMAP2": embeddings[:, 1],
        "UMAP3": embeddings[:, 2],
        label_name: labels.values
    })

    fig = go.Figure()

    # Plot label=1 first so it’s drawn on top
    for label_value in sorted(df_plot[label_name].unique(), reverse=True):
        df_sub = df_plot[df_plot[label_name] == label_value]
        fig.add_trace(go.Scatter3d(
            x=df_sub["UMAP1"],
            y=df_sub["UMAP2"],
            z=df_sub["UMAP3"],
            mode="markers",
            marker=dict(
                size=4,
                opacity=0.6
            ),
            name=f"{label_name}={label_value}"
        ))

    fig.update_layout(
        title=f"3D UMAP Plot by {label_name}",
        scene=dict(
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            zaxis_title="UMAP3"
        ),
        width=900,
        height=700,
        legend=dict(itemsizing='constant')
    )

    fig.show()