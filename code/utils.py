import os
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from promptsource.templates import DatasetTemplates
from promptsource.templates import TemplateCollection

import torch
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, TensorDataset, DataLoader


class NLIDataset(Dataset):
    def __init__(self, data_name, data, tokenizer, size=50000, content='all'):
        self.data_name = data_name
        self.data_size = min(len(data['premise']), size)
        self.premise = data['premise'][:self.data_size]
        self.hypothesis = data['hypothesis'][:self.data_size]
        self.premise = [[sent] for sent in self.premise]
        self.hypothesis = [[sent] for sent in self.hypothesis]
        self.label = [lab if lab >= 0 else 1 for lab in data['label']][:self.data_size]
        self.content = content
        
        self.tokenizer = tokenizer
        self.train_data = None
        self.init_data()
    
    def init_data(self):
        self.train_data = self.prep_data()
    
    def prep_data(self):
        MAX_LEN = 128
        if self.data_name == 'xnli':
            FLAG = True
        else:
            FLAG = False
        input_ids = torch.zeros(self.data_size, MAX_LEN, dtype=torch.long)
        token_type_ids = torch.zeros(self.data_size, MAX_LEN, dtype=torch.long)
        attention_mask = torch.zeros(self.data_size, MAX_LEN, dtype=torch.long)
        for i, (pre, hyp) in enumerate(zip(tqdm(self.premise), self.hypothesis)):
            if FLAG: 
                languages= hyp[0]['language']
                for j, lang in enumerate(languages):
                    cur_pre = pre[0][lang]
                    cur_hyp = hyp[0]['translation'][j]
                    idx = i * 15 + j
                    if idx == self.data_size:
                        break
                    if self.content == 'all':
                        input_ids[idx], token_type_ids[idx], attention_mask[idx] = self.tokenizer(cur_pre, cur_hyp, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt').values()
                    elif self.content == 'hypothesis':
                        input_ids[idx], token_type_ids[idx], attention_mask[idx] = self.tokenizer(cur_hyp, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt').values()
            else:
                idx = 0
                if self.content == 'all':
                    input_ids[i], token_type_ids[i], attention_mask[i] = self.tokenizer(pre, hyp, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt').values()
                elif self.content == 'hypothesis':
                    input_ids[i], token_type_ids[i], attention_mask[i] = self.tokenizer(hyp, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt').values()
                elif self.content == 'premise':
                    input_ids[i], token_type_ids[i], attention_mask[i] = self.tokenizer(pre, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt').values()
            if idx == self.data_size:
                print(1)
                break
        
        dataset = TensorDataset(input_ids, token_type_ids, attention_mask, torch.tensor(self.label))
        return dataset

    def get_data_loaders(self, batch_size, shuffle=True):
        train_loader = DataLoader(
        self.train_data,
        shuffle=shuffle,
        batch_size=batch_size
        )
        return train_loader 


def check_subset(dataset):
    collection = TemplateCollection().datasets_templates

    subset_options = []
    for data_name, subset_name in collection.keys():
        if data_name == dataset:
            subset_options.append(subset_name)

    SELECTED_SUBSET_NAME = None
    if subset_options:
        print(f"\nChoose a subset of {dataset}:\n")
        for idx, option in enumerate(subset_options):
            print(str(idx) + '. ' + option)
        select_idx = input("\nSelect the index: ")
        select_idx = 0
        SELECTED_SUBSET_NAME = subset_options[int(select_idx)]

    return SELECTED_SUBSET_NAME


def check_prompt_number(prompt_names, dataset):
    if len(prompt_names) > 1:
        print(f"\nChoose a prompt of {dataset}:\n")
        for idx, option in enumerate(prompt_names):
            print(str(idx) + '. ' + option)
        select_idx = 0
        SELECTED_PROMPT_NAME = prompt_names[int(select_idx)]
        return SELECTED_PROMPT_NAME
    else:
        return None


def get_name(content):
    if content == 'premise':
        return 'state_dict_nli_premise_only'
    elif content == 'hypothesis':
        return 'state_dict_nli_hypothesis_only'
    else:
        return 'state_dict_nli'
    
    
def load_data(data_name):
    templates = DatasetTemplates(data_name)
    prompt_names = templates.all_template_names
    SELECTED_PROMPT_NAME = check_prompt_number(prompt_names, data_name)
    if SELECTED_PROMPT_NAME:
        templates = templates[SELECTED_PROMPT_NAME]
    if data_name == 'xnli':
        dataset = load_dataset(data_name, 'all_languages')
    else:
        dataset = load_dataset(data_name)

    return dataset, templates

def set_seed():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)