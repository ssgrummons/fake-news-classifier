from transformers import AutoTokenizer, DataCollatorWithPadding

label2id = {'dickens': 0,
              'doyle': 1,
              'twain': 2,
              'defoe': 3}
id2label = {v: k for k, v in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# Pad the dataset to a fixed length.  
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

__all__ = ["tokenizer", "label2id", "id2label"]
