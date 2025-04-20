from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
import os

from src.data_utils import load_data
from src.dataset import IMDbDataset

# Load and prepare data
texts, labels = load_data("data/train.csv")
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenization
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# Dataset
dataset_train = IMDbDataset(train_encodings, train_labels)
dataset_val = IMDbDataset(val_encodings, val_labels)

# Model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="outputs/logs",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    logging_dir="outputs/logs",
    logging_steps=50,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val
)

trainer.train()

# Save model
os.makedirs("model", exist_ok=True)
model.save_pretrained("model")
tokenizer.save_pretrained("model")