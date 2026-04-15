import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import traceback, sys

# ----------------------
# Load dataset
# ----------------------
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
train_df["label"] = train_df["label"].astype(int)
val_df["label"] = val_df["label"].astype(int)

class NewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        text = str(self.df.loc[index, "text"])
        label = int(self.df.loc[index, "label"])
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len)
        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# ----------------------
# Tokenizer + Model
# ----------------------
print("Loading tokenizer and model (may download if not cached)...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

train_dataset = NewsDataset(train_df, tokenizer)
val_dataset = NewsDataset(val_df, tokenizer)

# ----------------------
# TrainingArguments (robust)
# ----------------------
OUTDIR = "./distil_model"

def make_args_with_fallback():
    # Primary desired settings
    kwargs = dict(
        output_dir=OUTDIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_steps=10,
    )
    try:
        print("Attempting to create TrainingArguments with modern kwargs...")
        ta = TrainingArguments(**kwargs)
        return ta
    except TypeError as e:
        # fallback for older transformers - remove unknown kwargs
        print("TrainingArguments init failed with TypeError (older transformers?). Falling back to safe minimal args.")
        # print traceback for debug
        traceback.print_exc()
        safe = dict(
            output_dir=OUTDIR,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=2e-5,
            logging_dir=OUTDIR + "/logs"
        )
        ta = TrainingArguments(**safe)
        return ta

training_args = make_args_with_fallback()

# ----------------------
# Trainer
# ----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# ----------------------
# TRAIN!
# ----------------------
print("Starting training...")
trainer.train()
print("Training finished. Saving model/tokenizer to", OUTDIR)
trainer.save_model(OUTDIR)
tokenizer.save_pretrained(OUTDIR)
print("Saved. Done.")
