from lexicalrichness import LexicalRichness
import pandas as pd

def compute_lexical_metrics(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    def metrics(text):
        try:
            lex = LexicalRichness(text)
            if lex.words == 0:
                return pd.Series({
                    "ttr": None,
                    "mtld": None,
                    "hdd": None,
                    "terms": 0
                })
            hdd_val = None
            try:
                hdd_val = lex.hdd(draws=min(42, max(1, int(0.8 * lex.words))))
            except Exception:
                pass  # Some texts may still throw in hdd
            return pd.Series({
                "ttr": lex.ttr,
                "mtld": lex.mtld(),
                "hdd": hdd_val,
                "terms": lex.terms
            })
        except Exception as e:
            # If LexicalRichness itself fails, catch it
            return pd.Series({
                "ttr": None,
                "mtld": None,
                "hdd": None,
                "terms": 0
            })

    metrics_df = df[text_col].apply(metrics)
    return pd.concat([df, metrics_df], axis=1)



def group_stats(df: pd.DataFrame, group_col: str = "is_bs", metrics=["ttr", "mtld", "hdd"]):
    return df.groupby(group_col)[metrics].describe()
