# from transformers import BertForSequenceClassification
# from transformers import AdamW
# from transformers import BertTokenizer

# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
# model.train()
# optimizer = AdamW(model.parameters(), lr=1e-5)

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# text_batch = ["I love Pixar.", "I don't care for Pixar."]
# encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
# input_ids = encoding['input_ids']
# attention_mask = encoding['attention_mask']

# labels = torch.tensor([1,0]).unsqueeze(0)
# outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
# loss = outputs.loss
# loss.backward()
# optimizer.step()

  
import os
import torch
from torch.utils import data
from PIL import Image
import cv2
import json
import numpy as np

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_path, transform, train=True):
        'Initialization'
        fin = open(list_path, 'r')
        self.meta = fin.readlines()
        fin.close()
        print('load meta: ', list_path)
        self.transform = transform
        self.train = train

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.meta)

    def __getitem__(self, index):
        'Generates one sample of data'
        image_path, label = self.meta[index].split()
        label = int(label)
        image = Image.open(image_path) 
        if self.transform is not None:
            sample = self.transform(image)
        if self.train:
            return sample, label
        else:
            return image_path, sample, label


from transformers import BertForSequenceClassification, Trainer, TrainingArguments

#model = BertForSequenceClassification.from_pretrained("bert-large-uncased")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total # of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset            # evaluation dataset
)

trainer.train()
trainer.evaluate()
model.save_pretrained(save_directory)