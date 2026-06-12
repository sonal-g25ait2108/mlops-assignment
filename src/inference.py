import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = os.getenv("HF_MODEL_NAME")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return pred

if __name__ == "__main__":
    text = os.getenv("INPUT_TEXT", "Hello world")
    print("Input:", text)
    print("Prediction:", predict(text))