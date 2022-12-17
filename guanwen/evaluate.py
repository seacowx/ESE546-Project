# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 21:30:22 2022

@author: nayut
"""
import torch
from torch.nn.functional import softmax
def prediction(model, data_loader, device, adv=False):
    model.eval()
    preds_list = []
    labels_list = []
    model.to(device)
    with torch.no_grad():
        for premise, hypothesis, labels in data_loader:
            premise = (premise[0].to(device), premise[1].to(device), premise[2].to(device))
            hypothesis = (hypothesis[0].to(device), hypothesis[1].to(device), hypothesis[2].to(device))
            labels = labels.to(device)
            if adv:
                pred = softmax(model(premise=premise, hypothesis=hypothesis)[0], dim=1)
            else:
                pred = softmax(model(premise=premise, hypothesis=hypothesis), dim=1)
            
            preds_list.append(pred)
            labels_list.append(labels)
            
    return torch.concat(preds_list), torch.concat(labels_list)
    
def prediction_hypothesis_cls(model, data_loader, device):
    model.eval()
    preds_list = []
    labels_list = []
    model.to(device)
    with torch.no_grad():
        for premise, hypothesis, labels in data_loader:
            hypothesis = (hypothesis[0].to(device), hypothesis[1].to(device), hypothesis[2].to(device))
            labels = labels.to(device)
            pred = softmax(model(hypothesis), dim=1)
            
            preds_list.append(pred)
            labels_list.append(labels)
            
    return torch.concat(preds_list), torch.concat(labels_list)