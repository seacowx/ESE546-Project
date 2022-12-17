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
#dataset name can be hans or snli
dataset_name = "hans"
dataset = load_dataset(dataset_name)
dataset = concatenate_datasets([dataset["train"], dataset["validation"]]).filter(lambda x: x["label"] != -1)
REQUIRED_LENGTH=9999999
#Change to bert tokenizer if using bert
#TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
TOKENIZER = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
hans_dataset = NLIDataset(dataset["premise"][:REQUIRED_LENGTH], 
                          dataset["hypothesis"][:REQUIRED_LENGTH],
                          dataset["label"][:REQUIRED_LENGTH], TOKENIZER,split=True)
hans_dataloader = DataLoader(hans_dataset, batch_size=64, shuffle=True)

model =  AdversialSameDistilBertClassifier(0)
#ToChange
model_state_dict_path = "./state_dicts/adv_one_encoder_state_dict_67.pt"
model.load_state_dict(torch.load(model_state_dict_path))

#ToChange if using adv, adv=True
predictions, labels = prediction(model, hans_dataloader, device, adv=True)
hans_id_2_labels = {0:"entailment", 1:"non_entailment"}
combined_predictions = torch.zeros(predictions.shape[0], 2)
combined_predictions[:,0] = predictions[:,0]
combined_predictions[:,1] = predictions[:,1] + predictions[:,2]
print(predictions[5000:])
acc = accuracy(predictions, labels)
print(labels.unique(return_counts=True))
print(acc)