import pandas as pd
import nltk
import re
from collections import defaultdict
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

def compute_length(text: str, method: str = 'chars') -> int:
    """
    Compute the length of a text based on the specified method.

    Args:
        text (str): Text to compute the length for.
        method (str, optional): Method to use for computing the length. Can be 'chars', 'words', or 'tokens'. Defaults to 'chars'.

    Raises:
        ValueError: If the method is not one of 'chars', 'words', or 'tokens'.

    Returns:
        int: Counter of characters, words, or tokens in the text.
    """
    if method == 'chars':
        return len(text)
    elif method == 'words':
        return len(text.split())
    elif method == 'tokens':
        return len(word_tokenize(text))
    else:
        raise ValueError("method must be one of: 'chars', 'words', 'tokens'")
    
def add_length_column(df: pd.DataFrame, text_col: str = 'text', method: str = 'chars') -> pd.DataFrame:
    """
    Add a new column to the DataFrame that contains the computed length of the text based on the specified method.

    Args:
        df (pd.DataFrame): DataFrame with a text column.
        text_col (str, optional): Column name containing the text to compute the length for. Defaults to 'text'.
        method (str, optional): Method to use for computing the length. Can be 'chars', 'words', or 'tokens'. Defaults to 'chars'.

    Returns:
        pd.DataFrame: DataFrame with the new column added. The new column is named after the method used for computing the length.
    """
    df = df.copy()
    length_col = f'{method}_length'
    df[length_col] = df[text_col].apply(lambda x: compute_length(x, method))
    return df, length_col

def describe_lengths(df: pd.DataFrame, length_col: str, group_col: str = 'is_bs') -> pd.DataFrame:
    """
    Returns a DataFrame with descriptive statistics for the computed lengths grouped by a specified column.

    Args:
        df (pd.DataFrame): Dataframe with the computed length column.
        length_col (str): Name of the computed length column.
        group_col (str, optional): Name of the column to group by. Defaults to 'author'.

    Returns:
        pd.DataFrame: Description statistics for the computed lengths grouped by the specified column.
    """
    return df.groupby(group_col)[length_col].describe()


def detect_author_name_leakage(df: pd.DataFrame, text_col: str, label_col: str, author_aliases: dict):
    """
    Detects whether any author's name or known aliases appear in their associated text entries.

    Args:
        df (pd.DataFrame): Input dataframe.
        text_col (str): Name of the text column.
        label_col (str): Name of the label column (e.g. 'author').
        author_aliases (dict): Dictionary where keys are author labels, and values are lists of name variants.

    Returns:
        dict: Count of potential leakage cases for each author.
    """
    leakage_counts = defaultdict(int)

    for _, row in df.iterrows():
        text = row[text_col].lower()
        label = row[label_col]
        for alias in author_aliases.get(label, []):
            if re.search(r'\b' + re.escape(alias.lower()) + r'\b', text):
                leakage_counts[label] += 1
                break  # Don't count multiple hits per sample

    return dict(leakage_counts)
