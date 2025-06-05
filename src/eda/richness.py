from lexicalrichness import LexicalRichness
import pandas as pd

def compute_lexical_metrics(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    def metrics(text):
        lex = LexicalRichness(text)
        try:
            hdd_val = lex.hdd(draws=min(42, max(1, int(0.8 * lex.words))))
        except Exception:
            hdd_val = None
        return pd.Series({
            "ttr": lex.ttr,
            "mtld": lex.mtld() if lex.words > 0 else None,
            "hdd": hdd_val,
            "terms": lex.terms
        })

    metrics_df = df[text_col].apply(metrics)
    return pd.concat([df, metrics_df], axis=1)


def group_stats(df: pd.DataFrame, group_col: str = "author", metrics=["ttr", "mtld", "hdd"]):
    return df.groupby(group_col)[metrics].describe()
