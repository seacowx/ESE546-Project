# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 18:16:48 2022

@author: nayut
"""

from NLIDataset import NLIDataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from train import train_hypothesis_cls
from datasets import load_dataset, concatenate_datasets
from evaluate import prediction
from transformers import BertTokenizer, DistilBertTokenizer
from models import BasicDoubleDistilBertClassifier, BasicSingleDistilBertClassifier, \
    SameDistilBertClassifier, AdversialSameDistilBertClassifier, HypothesisLabelClassifier
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
#dataset name can be hans or snli
dataset_name = "snli"
dataset = load_dataset(dataset_name)
dataset = concatenate_datasets([dataset["train"], dataset["validation"]]).filter(lambda x: x["label"] != -1)
REQUIRED_LENGTH=99999
#Change to bert tokenizer if using bert
#TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
snli = load_dataset("snli")
snli_train = concatenate_datasets([snli["train"], snli["validation"]]).filter(lambda x: x["label"] != -1)
snli_test = snli["test"].filter(lambda x: x["label"] != -1)
TRAINING_REQUIRED_LENGTH=50000
TESTING_REQUIRED_LENGTH=3000
#TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
TOKENIZER = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
train_dataset = NLIDataset(snli_train["premise"][:TRAINING_REQUIRED_LENGTH], snli_train["hypothesis"][:TRAINING_REQUIRED_LENGTH],snli_train["label"][:TRAINING_REQUIRED_LENGTH], TOKENIZER,split=True)
test_dataset = NLIDataset(snli_test["premise"][:TESTING_REQUIRED_LENGTH], snli_test["hypothesis"][:TESTING_REQUIRED_LENGTH],snli_test["label"][:TESTING_REQUIRED_LENGTH], TOKENIZER, split=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model =  AdversialSameDistilBertClassifier(0)
#ToChange
model_state_dict_path = "./state_dicts/best_acc_68_distilbert_one_encoder_two_embed.pt"
model.load_state_dict(torch.load(model_state_dict_path))
encoder = model.g
model = HypothesisLabelClassifier(encoder)
#ToChange if using adv, adv=True

n_epochs = 10
lr = 0.00001
data_name = "snli"
context = "test1"
report_period = 50
criterion = nn.CrossEntropyLoss()
#adv_model = AdversialClassifier(alpha)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
decay_lambda = lambda epoch: 0.97 ** epoch
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = decay_lambda)

train_hypothesis_cls(model, train_loader, test_loader, optimizer, 
                  n_epochs, data_name, context, device, criterion, lr_scheduler,
                  report_period=50)