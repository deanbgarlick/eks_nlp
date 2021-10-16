import pandas as pd
import torch

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

from dataset import DisasterTweetsDataset


train = pd.read_csv('data/train.csv')

def preprocess_row(x):
    return x.text + ' [KEYWORD] ' + str(x.keyword) + ' [LOCATION] ' + str(x.location)

train['tweet'] = train.apply(preprocess_row, axis=1)

train_tweets, train_labels = train.tweet.astype(str).tolist(), train.target.astype(int).tolist()
train_tweets, val_tweets, train_labels, val_labels = train_test_split(train_tweets, train_labels, test_size=.2)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(train_tweets, truncation=True, padding=True)
val_encodings = tokenizer(val_tweets, truncation=True, padding=True)

train_dataset = DisasterTweetsDataset(train_encodings, train_labels)
val_dataset = DisasterTweetsDataset(val_encodings, val_labels)


training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)



trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()

tokenizer.save_pretrained('binaries/tokenizer')
model.save_pretrained('binaries/model')
