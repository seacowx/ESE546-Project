# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 11:53:54 2022

@author: nayut
"""
import torch
from torch import nn
from torch.autograd import Function
from transformers import  BertModel, DistilBertModel
BERT_OUTPUT_SIZE = 768

class BasicSingleDistilBertClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    # output size in (batch_size, seq_len, 768)
    self.g = DistilBertModel.from_pretrained("distilbert-base-uncased")
    self.linear = nn.Linear(BERT_OUTPUT_SIZE, 3)

  def forward(self, tokens):
    re = self.g(input_ids=tokens[0], attention_mask=tokens[2])["last_hidden_state"][:,0,:]
    #print(concated.shape)
    o= self.linear(re)
    #print(o.shape)
    return o

class SameDistilBertClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    # output size in (batch_size, seq_len, 768)
    self.g = DistilBertModel.from_pretrained("distilbert-base-uncased")
    self.cf = nn.Sequential(
        nn.Linear(4*BERT_OUTPUT_SIZE, BERT_OUTPUT_SIZE),
        nn.Dropout(0.5),
        nn.BatchNorm1d(BERT_OUTPUT_SIZE),
        nn.Linear(BERT_OUTPUT_SIZE,BERT_OUTPUT_SIZE//8),
        nn.Dropout(0.5),
        nn.BatchNorm1d(BERT_OUTPUT_SIZE//8),
        nn.Linear(BERT_OUTPUT_SIZE//8, 3),
    )

  def forward(self, premise, hypothesis):
    premise_re = self.g(input_ids=premise[0],attention_mask=premise[2])["last_hidden_state"][:,0,:]
    hypothesis_re = self.g(input_ids=hypothesis[0], attention_mask=hypothesis[2])["last_hidden_state"][:,0,:]
    concated = torch.concat([premise_re-hypothesis_re, premise_re*hypothesis_re, premise_re, hypothesis_re], dim=1)
    #print(concated.shape)
    o= self.cf(concated)
    #print(o.shape)
    return o

class BasicDoubleDistilBertClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    # output size in (batch_size, seq_len, 768)
    self.gh = DistilBertModel.from_pretrained("distilbert-base-uncased")
    self.gp = DistilBertModel.from_pretrained("distilbert-base-uncased")
    self.cf = nn.Sequential(
        nn.Linear(4*BERT_OUTPUT_SIZE, BERT_OUTPUT_SIZE),
        nn.Dropout(0.5),
        nn.BatchNorm1d(BERT_OUTPUT_SIZE),
        nn.Linear(BERT_OUTPUT_SIZE,BERT_OUTPUT_SIZE//8),
        nn.Dropout(0.5),
        nn.BatchNorm1d(BERT_OUTPUT_SIZE//8),
        nn.Linear(BERT_OUTPUT_SIZE//8, 3),
    )

  def forward(self, premise, hypothesis):
    premise_re = self.gp(input_ids=premise[0],attention_mask=premise[2])["last_hidden_state"][:,0,:]
    hypothesis_re = self.gh(input_ids=hypothesis[0], attention_mask=hypothesis[2])["last_hidden_state"][:,0,:]
    concated = torch.concat([premise_re-hypothesis_re, premise_re*hypothesis_re, premise_re, hypothesis_re], dim=1)
    #print(concated.shape)
    o= self.cf(concated)
    #print(o.shape)
    return o

class BasicDoubleBertClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    # output size in (batch_size, seq_len, 768)
    self.gh = BertModel.from_pretrained("bert-base-uncased")
    self.gp = BertModel.from_pretrained("bert-base-uncased")
    self.linear = nn.Linear(2*BERT_OUTPUT_SIZE, 3)

  def forward(self, premise, hypothesis):
    premise_re = self.gp(input_ids=premise[0], token_type_ids=premise[1], attention_mask=premise[2])["last_hidden_state"][:,0,:]
    hypothesis_re = self.gh(input_ids=hypothesis[0], token_type_ids=hypothesis[1], attention_mask=hypothesis[2])["last_hidden_state"][:,0,:]
    concated = torch.concat([premise_re, hypothesis_re], dim=1)
    #print(concated.shape)
    o= self.linear(concated)
    #print(o.shape)
    return o

class BasicOneBertClassifier(nn.Module):
    def __init__(self):
      super().__init__()
      # output size in (batch_size, seq_len, 768)
      self.g = BertModel.from_pretrained("bert-base-uncased")
      self.linear = nn.Linear(BERT_OUTPUT_SIZE, 3)

    def forward(self, tokens):
      re = self.g(input_ids=tokens[0], token_type_ids=tokens[1], attention_mask=tokens[2])["last_hidden_state"][:,0,:]
      #print(concated.shape)
      o= self.linear(re)
      #print(o.shape)
      return o
  
class GradientReversal(Function):
  @staticmethod
  def forward(ctx, x, alpha):
    # Doing Nothing
    ctx.save_for_backward(x, alpha)
    return x
  
  def backward(ctx, grad_output):
    #https://github.com/tadeephuy/GradientReversal/blob/master/gradient_reversal/functional.py
    #input number should be the same as the output number of forward
    #output number should be same as the input of the forward
    grad_input = None
    _, alpha = ctx.saved_tensors
    #condition can be deleted?
    if ctx.needs_input_grad[0]:
      grad_input = -alpha*grad_output
    return grad_input, None
  
revgrad = GradientReversal.apply

class GradientReversalLayer(nn.Module):
  def __init__(self, alpha):
    super().__init__()
    self.alpha = torch.tensor(alpha, requires_grad=False)
  
  def forward(self, x):
    return revgrad(x, self.alpha)


class AdversialSameDistilBertClassifier(nn.Module):
  def __init__(self, alpha):
    super().__init__()
    self.bias_classifier = nn.Sequential(
        nn.Linear(BERT_OUTPUT_SIZE, BERT_OUTPUT_SIZE//8),
        nn.Dropout(0.5),
        nn.BatchNorm1d(BERT_OUTPUT_SIZE//8),
        nn.Linear(BERT_OUTPUT_SIZE//8,BERT_OUTPUT_SIZE//16),
        nn.Dropout(0.5),
        nn.BatchNorm1d(BERT_OUTPUT_SIZE//16),
        nn.Linear(BERT_OUTPUT_SIZE//16, 3),
    )

    self.g = DistilBertModel.from_pretrained("distilbert-base-uncased")
    self.cf = nn.Sequential(
        nn.Linear(4*BERT_OUTPUT_SIZE, BERT_OUTPUT_SIZE),
        nn.Dropout(0.5),
        nn.BatchNorm1d(BERT_OUTPUT_SIZE),
        nn.Linear(BERT_OUTPUT_SIZE,BERT_OUTPUT_SIZE//8),
        nn.Dropout(0.5),
        nn.BatchNorm1d(BERT_OUTPUT_SIZE//8),
        nn.Linear(BERT_OUTPUT_SIZE//8, 3),
    )
    self.gradient_reversal = GradientReversalLayer(alpha)

  def forward(self, premise, hypothesis):
    
    premise_re = self.g(input_ids=premise[0], attention_mask=premise[2])["last_hidden_state"][:,0,:]
    hypothesis_re = self.g(input_ids=hypothesis[0], attention_mask=hypothesis[2])["last_hidden_state"][:,0,:]
    concated = torch.concat([premise_re-hypothesis_re, premise_re*hypothesis_re, premise_re, hypothesis_re], dim=1)
    label_prediction_combined = self.cf(concated)
    label_prediction_hypothesis = self.bias_classifier(self.gradient_reversal(hypothesis_re))
    return label_prediction_combined, label_prediction_hypothesis