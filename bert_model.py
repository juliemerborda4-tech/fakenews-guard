from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def bert_predict(text):
    result = classifier(text)[0]

    label = result["label"]
    score = result["score"]

    if label == "POSITIVE":
        return "real", score
    else:
        return "fake", score