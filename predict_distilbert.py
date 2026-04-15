# predict_distilbert.py
import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_DIR = "distil_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

def predict_with_distilbert(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    padding=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        logits = out.logits.cpu().numpy()[0]

    # stable softmax
    mx = np.max(logits)
    exps = np.exp(logits - mx)
    probs = exps / exps.sum()

    pred_idx = int(np.argmax(probs))
    label = "fake" if pred_idx == 1 else "real"
    confidence = float(probs[pred_idx])

    return label, confidence, {"real": float(probs[0]), "fake": float(probs[1])}

if __name__ == "__main__":
    s = "Breaking: DSWD, DILG sign program to help street dwellers"
    lab, conf, probs = predict_with_distilbert(s)
    print("LABEL:", lab)
    print("CONF:", conf)
    print("PROBS:", probs)
