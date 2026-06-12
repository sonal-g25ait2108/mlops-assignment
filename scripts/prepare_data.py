import pandas as pd
import json
import re
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
RAW_DATA = "data/raw/SMSSpamCollection"
PROCESSED_DATA = "data/processed/cleaned_sms_spam.csv"
LABEL_FILE = "data/id2label.json"

# Create output directory
Path("data/processed").mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(
    RAW_DATA,
    sep="\t",
    header=None,
    names=["label", "text"]
)

# -----------------------------
# Dataset Inspection
# -----------------------------
print("\n===== DATASET INSPECTION =====")
print(f"Dataset Shape: {df.shape}")

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df["label"].value_counts())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# -----------------------------
# Data Cleaning
# -----------------------------

# Remove missing values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

df["text"] = df["text"].apply(clean_text)

# -----------------------------
# Encode Labels
# -----------------------------
label2id = {
    "ham": 0,
    "spam": 1
}

id2label = {
    "0": "ham",
    "1": "spam"
}

df["label"] = df["label"].map(label2id)

# -----------------------------
# Save Label Mapping
# -----------------------------
with open(LABEL_FILE, "w") as f:
    json.dump(id2label, f, indent=4)

# -----------------------------
# Save Processed Dataset
# -----------------------------
df.to_csv(PROCESSED_DATA, index=False)

# -----------------------------
# Final Summary
# -----------------------------
print("\n===== CLEANING COMPLETE =====")
print(f"Final Dataset Shape: {df.shape}")

print("\nEncoded Class Distribution:")
print(df["label"].value_counts())

print(f"\nProcessed dataset saved to:")
print(PROCESSED_DATA)

print(f"\nid2label mapping saved to:")
print(LABEL_FILE)
