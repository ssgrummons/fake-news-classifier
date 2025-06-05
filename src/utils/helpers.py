from sklearn.metrics import classification_report
import pandas as pd
from IPython.display import display, Markdown

def classification_report_to_df(y_true, y_pred, digits=2) -> pd.DataFrame:
    """
    Generate a classification report as a pandas DataFrame.

    Args:
        y_true (List or np.ndarray): True labels.
        y_pred (List or np.ndarray): Predicted labels.
        digits (int): Number of decimal digits to round.

    Returns:
        pd.DataFrame: Formatted classification report.
    """
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    
    # Clean up float precision and cast support to int
    for col in ['precision', 'recall', 'f1-score']:
        if col in df.columns:
            df[col] = df[col].round(digits)
    if 'support' in df.columns:
        df['support'] = df['support'].astype(int)
    
    return df

def md_print(text: str):
    display(Markdown(text))