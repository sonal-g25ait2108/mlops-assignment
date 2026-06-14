# mlops-assignment
# MLOps Assignment 3 – SMS Spam Classification

## Overview

This project implements an end-to-end MLOps pipeline for SMS spam classification using Hugging Face Transformers, Kaggle, Weights & Biases (W&B), Docker, and GitHub Actions.

A DistilBERT-based model is fine-tuned on the SMS Spam Collection dataset to classify messages as either:

* **Ham (0)** – legitimate messages
* **Spam (1)** – unwanted promotional or fraudulent messages

The project demonstrates data preparation, model training, experiment tracking, model deployment, containerization, and CI/CD automation.

---

## Dataset

Dataset: SMS Spam Collection

The dataset contains SMS messages labeled as:

* Ham
* Spam

After preprocessing, labels were converted to numeric values:

| Label | Class |
| ----- | ----- |
| 0     | Ham   |
| 1     | Spam  |

---

## Model

Model used:

**DistilBERT for Sequence Classification**

Hugging Face Model:

`distilbert-base-uncased`

The model was selected because it provides strong text classification performance while remaining lightweight enough to train efficiently within Kaggle's free GPU limits.

---

## Experiment Tracking

Training experiments were tracked using **Weights & Biases (W&B)**.

Metrics logged:

* Training Loss
* Validation Loss
* Accuracy
* F1 Score

Multiple experiment versions were trained using different hyperparameters and compared through the W&B dashboard.

Run inference:

```bash
docker run --rm \
  -e HF_TOKEN=<your_hf_token> \
  -e INPUT_TEXT="Free entry win prize!!!" \
  mlops-a3-inference:latest
```

Example output:

```text
Input: Free entry win prize!!!
Prediction: spam
```

---

## GitHub Actions

### CI Workflow

Triggered on:

* Push to `develop`
* Pull requests to `main`

Checks:

* Dependency installation
* Flake8 linting

### Inference Workflow

Triggered manually through GitHub Actions.

Input:

```text
Text to classify
```

Output:

```text
Prediction: ham/spam
```

---

## Repository Structure

```text
.
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── inference.yml
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── load_model.py
│   └── prepare_data.py
├── src/
│   └── inference.py
├── Dockerfile
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Technologies Used

* Python
* Hugging Face Transformers
* Hugging Face Hub
* Kaggle Notebooks
* Weights & Biases (W&B)
* Docker
* GitHub Actions
* Scikit-learn
* PyTorch

---

## Author

Sonal Bajaj

MLOps Assignment 3
