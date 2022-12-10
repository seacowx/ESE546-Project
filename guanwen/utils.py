# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 11:50:25 2022

@author: nayut
"""
import torch
def accuracy(pred, labels):
  pred = torch.argmax(pred, dim=1)
  return torch.sum(pred==labels)/labels.shape[0]

def correct_num(pred, labels):
    pred = torch.argmax(pred, dim=1)
    return torch.sum(pred==labels)