# import os
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

# model_name = os.getenv("HF_MODEL_NAME")

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

# def predict(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     outputs = model(**inputs)
#     pred = torch.argmax(outputs.logits, dim=1).item()
#     return pred

# if __name__ == "__main__":
#     text = os.getenv("INPUT_TEXT", "Hello world")
#     print("Input:", text)
#     print("Prediction:", predict(text))


import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "sonalbajaj/sms-spam-distilbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

text = os.environ["INPUT_TEXT"]

inputs = tokenizer(text, return_tensors="pt", truncation=True)

outputs = model(**inputs)

prediction = torch.argmax(outputs.logits, dim=1).item()

label_map = {
    0: "ham",
    1: "spam"
}

print("Input:", text)
print("Prediction:", label_map[prediction])
