FROM python:3.10-slim

WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy code
COPY . .

# build argument for HF model
ARG HF_MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
ENV HF_MODEL_NAME=$HF_MODEL_NAME

# default command
CMD ["python", "inference.py"]