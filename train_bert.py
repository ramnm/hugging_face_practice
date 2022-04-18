import numpy as np
from datasets import load_dataset, load_metric
from transformers import DefaultDataCollator
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
import tensorflow as tf


dataset = load_dataset("emotion")
dataset["train"][100]


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=6)
training_args = TrainingArguments(output_dir="test_trainer")
metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)


trainer.train()

outputs = trainer.predict(small_eval_dataset)
y_pred = outputs.predictions.argmax(1)

y_pred[:10]
small_eval_dataset['text'][:10]

counter = -1
index_list = []
for i, j in zip(y_pred, small_eval_dataset['label']):
    counter += 1
    if i != j:
        index_list.append(counter)