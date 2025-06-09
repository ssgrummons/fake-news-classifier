from transformers import AutoTokenizer, DataCollatorWithPadding


tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# Pad the dataset to a fixed length.  
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

__all__ = ["tokenizer"]
