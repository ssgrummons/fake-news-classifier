from ftfy import fix_text
import re

def clean_text(text: str, 
               use_ftfy: bool = True, 
               lowercase: bool = True, 
               strip_whitespace: bool = True) -> str:
    """
    Normalize text using ftfy if enabled.
    """
    if use_ftfy:
        return fix_text(text)
    if lowercase:
        text = text.lower()
    if strip_whitespace:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
    return text
