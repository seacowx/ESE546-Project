# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 21:30:22 2022

@author: nayut
"""
import torch
from torch.nn.functional import softmax
from utils import accuracy
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
def accuracy_AdvDat(model, data_loader, device):
    model.eval()
    model.to(device)
    with torch.no_grad():
        val_acc_unperturb_all = 0
        val_total = 0
        for premise, hypothesis, labels, perturb_mask in data_loader:
            premise = (premise[0].to(device), premise[1].to(device), premise[2].to(device))
            hypothesis = (hypothesis[0].to(device), hypothesis[1].to(device), hypothesis[2].to(device))
            labels = labels.to(device)
            perturb_mask = perturb_mask.to(device)
            unperturb_pred, perturb_pred = model(premise=premise, hypothesis=hypothesis, perturb_mask=perturb_mask)
            if unperturb_pred != None:
                val_acc_unperturb = accuracy(unperturb_pred, labels[perturb_mask==0]).item()
                val_acc_unperturb_all += val_acc_unperturb 
                
            val_total += 1

            
        aver_acc_unperturb = val_acc_unperturb_all / val_total
        
            
    return aver_acc_unperturb
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