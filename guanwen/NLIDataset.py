# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 11:47:53 2022

@author: nayut
"""
from torch.utils.data import Dataset
import torch
import numpy as np
MAX_LENGTH = 128
class NLIDataset(Dataset):
  def __init__(self, premise, hypothesis, labels, tokenizer, 
               split=True, distill_bert=True, 
               label_perturb=False, fraction=None, shuffle=True):
    super().__init__()
    self.label_perturb = label_perturb
    if shuffle:
        premise, hypothesis, labels = np.array(premise), np.array(hypothesis),np.array(labels)
        random_indexes = torch.randperm(len(premise))
        premise = premise[random_indexes].tolist()
        hypothesis = hypothesis[random_indexes].tolist()
        labels = labels[random_indexes].tolist()
    if label_perturb:
        assert(not fraction is None)
        premise, hypothesis, labels = np.array(premise), np.array(hypothesis),np.array(labels)
        split_index = round(len(labels)*(1-fraction))
        #This is an inplace operation!
        perturbing_premise = premise[split_index:]
        premise[split_index:] = perturbing_premise[torch.randperm(len(perturbing_premise))]
        self.perturb_mask = torch.zeros(len(labels))
        self.perturb_mask[split_index:] = 1
        premise = premise.tolist()
        hypothesis = hypothesis.tolist()
        labels = labels.tolist()
    self.split = split
    self.distill_bert = distill_bert
    if self.split:
        premise = tokenizer(premise, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt")
        hypothesis = tokenizer(hypothesis, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt")
        self.premise_input_ids = premise["input_ids"]
        if not distill_bert:
            self.premise_token_type_ids = premise["token_type_ids"]
        self.premise_attention_mask = premise["attention_mask"]
    
        self.hypothesis_input_ids = hypothesis["input_ids"]
        if not distill_bert:
            self.hypothesis_token_type_ids = hypothesis["token_type_ids"]
        self.hypothesis_attention_mask = hypothesis["attention_mask"]
    else:
        tokens = tokenizer(premise, hypothesis, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt")
        
        self.concat_input_ids = tokens["input_ids"]
        if not distill_bert:
            self.concat_type_ids = hypothesis["token_type_ids"]
        self.concat_attention_mask = tokens["attention_mask"]
    self.labels = labels

  def __getitem__(self, idx):
    if self.split:
        if not self.distill_bert:
            res = [(self.premise_input_ids[idx], self.premise_token_type_ids[idx], self.premise_attention_mask[idx]),\
                (self.hypothesis_input_ids[idx], self.hypothesis_token_type_ids[idx], self.hypothesis_attention_mask[idx]),self.labels[idx]]
        else:
            res = [(self.premise_input_ids[idx], 0, self.premise_attention_mask[idx]),\
                (self.hypothesis_input_ids[idx], 0, self.hypothesis_attention_mask[idx]),self.labels[idx]]

    else:
        if not self.distill_bert:
            res = [(self.concat_input_ids[idx], self.concat_type_ids, self.concat_attention_mask[idx]), self.labels[idx]]
        else:
            res = [(self.concat_input_ids[idx], 0, self.concat_attention_mask[idx]), self.labels[idx]]
    if self.label_perturb:
        res.append(self.perturb_mask[idx])
    return res
  
  def __len__(self):
    return len(self.labels)