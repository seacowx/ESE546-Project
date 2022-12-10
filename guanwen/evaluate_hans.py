# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 21:15:44 2022

@author: nayut
"""
from NLIDataset import NLIDataset
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from evaluate import prediction
from utils import accuracy
from transformers import BertTokenizer, DistilBertTokenizer
from models import BasicDoubleDistilBertClassifier, BasicSingleDistilBertClassifier, \
    SameDistilBertClassifier, AdversialSameDistilBertClassifier
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
hans = load_dataset("hans")
hans = concatenate_datasets([hans["train"], hans["validation"]]).filter(lambda x: x["label"] != -1)
REQUIRED_LENGTH=3000
#TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
TOKENIZER = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
hans_dataset = NLIDataset(hans["premise"][:REQUIRED_LENGTH], 
                          hans["hypothesis"][:REQUIRED_LENGTH],
                          hans["label"][:REQUIRED_LENGTH], TOKENIZER,split=True)
hans_dataloader = DataLoader(hans_dataset, batch_size=64, shuffle=True)

model =  AdversialSameDistilBertClassifier(0)
model.load_state_dict(torch.load("best_acc_68_distilbert_one_encoder_two_embed.pt"))
predictions, labels = prediction(model, hans_dataloader, device, adv=True)
hans_id_2_labels = {0:"entailment", 1:"non_entailment"}
combined_predictions = torch.zeros(predictions.shape[0], 2)
combined_predictions[:,0] = predictions[:,0]
combined_predictions[:,1] = predictions[:,1] + predictions[:,2]
acc = accuracy(predictions, labels)
print(acc)