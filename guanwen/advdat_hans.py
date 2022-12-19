# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 17:56:31 2022

@author: nayut
"""

from NLIDataset import NLIDataset
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from evaluate import accuracy_AdvDat
from utils import accuracy
from transformers import BertTokenizer, DistilBertTokenizer
from models import BasicDoubleDistilBertClassifier, BasicSingleDistilBertClassifier, \
    SameDistilBertClassifier, AdversialSameDistilBertClassifier, AdvDatSameDistilBertClassifier
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
#dataset name can be hans or snli
dataset_name = "hans"
dataset = load_dataset(dataset_name)
dataset = concatenate_datasets([dataset["train"], dataset["validation"]]).filter(lambda x: x["label"] != -1)
REQUIRED_LENGTH=20000
#Change to bert tokenizer if using bert
#TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
TOKENIZER = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
hans_dataset = NLIDataset(dataset["premise"][:REQUIRED_LENGTH], 
                          dataset["hypothesis"][:REQUIRED_LENGTH],
                          dataset["label"][:REQUIRED_LENGTH], 
                          TOKENIZER,split=True, label_perturb=True,
                          fraction=0)
data_loader = DataLoader(hans_dataset, batch_size=64, shuffle=True)
model = AdvDatSameDistilBertClassifier(0)

model_state_dict_path = "./state_dicts/snli_adv_dat_perturb_0.4_alpha_0.4_state_dict.pt"
model.load_state_dict(torch.load(model_state_dict_path))

print(accuracy_AdvDat(model, data_loader, device))