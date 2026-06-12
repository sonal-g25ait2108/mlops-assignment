from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

MODEL_NAME = "distilbert-base-uncased"

# Load label mapping
with open("data/id2label.json", "r") as f:
    id2label = json.load(f)

label2id = {v: int(k) for k, v in id2label.items()}

num_labels = len(id2label)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label={int(k): v for k, v in id2label.items()},
    label2id=label2id
)

print("Tokenizer loaded successfully")
print("Model loaded successfully")
print(f"Number of labels: {num_labels}")
print("Labels:", id2label)