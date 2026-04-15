# train_model.py
import os
import math
import random
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Config
MODEL_NAME = "bert-base-uncased"   # change to multilingual if you need Tagalog/Filipino support
OUTPUT_DIR = "bert_model"
MAX_LEN = 128        # reduce token length (faster)
BATCH_SIZE = 8       # keep small for CPU
EPOCHS = 3           # 2–3 is fine for demo
LR = 2e-5
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
API_FEAT_NAMES = ["factcheck_bool", "num_gnews_hits", "top_source_reliability"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -- Dataset class ----------------------------------------------------------
class NewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256, api_feat_names=None):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.api_feat_names = api_feat_names or []
        # prepare api features
        self.api_feats = []
        for _, row in df.iterrows():
            feats = []
            for c in self.api_feat_names:
                try:
                    feats.append(float(row.get(c, 0.0)))
                except Exception:
                    feats.append(0.0)
            self.api_feats.append(feats)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "api_feats": torch.tensor(self.api_feats[idx], dtype=torch.float),
        }
        return item

# -- Model ------------------------------------------------------------------
class BertWithAPI(nn.Module):
    def __init__(self, base_model_name=MODEL_NAME, n_api_feats=3, hidden_dim=256):
        super().__init__()
        self.bert = BertModel.from_pretrained(base_model_name)
        bert_dim = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(bert_dim + n_api_feats, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, input_ids, attention_mask, api_feats):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = out.pooler_output  # [batch, hidden]
        if api_feats is None:
            api_feats = torch.zeros((pooled.size(0), 3), device=pooled.device)
        x = torch.cat([pooled, api_feats], dim=1)
        logits = self.classifier(x)
        return logits

# -- Utilities --------------------------------------------------------------
def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    api_feats = torch.stack([b["api_feats"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "api_feats": api_feats}

def compute_metrics(y_true, y_pred):
    preds = np.argmax(y_pred, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0))
    }

# -- Main training ----------------------------------------------------------
def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Create train.csv and val.csv first.")
    return pd.read_csv(path)

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Loading data...")
    train_df = load_csv("train.csv")
    val_df = load_csv("val.csv")

    # Ensure api feature columns exist
    for c in API_FEAT_NAMES:
        if c not in train_df.columns:
            train_df[c] = 0.0
        if c not in val_df.columns:
            val_df[c] = 0.0

    train_ds = NewsDataset(train_df, tokenizer, max_len=MAX_LEN, api_feat_names=API_FEAT_NAMES)
    val_ds = NewsDataset(val_df, tokenizer, max_len=MAX_LEN, api_feat_names=API_FEAT_NAMES)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = BertWithAPI(base_model_name=MODEL_NAME, n_api_feats=len(API_FEAT_NAMES)).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= int(0.1*total_steps), num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - train")
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            api_feats = batch["api_feats"].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, api_feats=api_feats)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} - val"):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                api_feats = batch["api_feats"].to(DEVICE)
                logits = model(input_ids=input_ids, attention_mask=attention_mask, api_feats=api_feats)
                loss = loss_fn(logits, labels)
                val_loss += loss.item()
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.cpu().numpy())
        all_preds = np.vstack(all_preds)
        all_labels = np.concatenate(all_labels)
        metrics = compute_metrics(all_labels, all_preds)
        avg_val_loss = val_loss / len(val_loader)

        print(f"\nEpoch {epoch} summary: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")
        print(f"VAL metrics: {metrics}")

        # Save best
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_path = os.path.join(OUTPUT_DIR, "model.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "tokenizer": tokenizer.save_pretrained(OUTPUT_DIR),
                "api_feat_names": API_FEAT_NAMES
            }, save_path)
            # tokenizer already saved above; save metadata
            meta = {"best_f1": best_f1}
            pd.Series(meta).to_json(os.path.join(OUTPUT_DIR, "meta.json"))
            print(f"Saved best model (f1={best_f1:.4f}) to {OUTPUT_DIR}")

    print("Training complete.")

if __name__ == "__main__":
    train()
