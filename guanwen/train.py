# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 11:50:54 2022

@author: nayut
"""
from utils import accuracy
import torch
import wandb
import numpy as np
def train_double_encoder(model, train_loader, test_loader, optimizer, 
                         n_epochs, data_name, context, device, criterion, lr_scheduler,
                         report_period=50, wandb=wandb, log=True):
    counter = 0
    best_val_acc = 0
    model = model.to(device)
    train_batchs_per_epoch = len(train_loader)
    for epoch in range(n_epochs):
        for batch ,(premise, hypothesis, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            premise = (premise[0].to(device), premise[1].to(device), premise[2].to(device))
            hypothesis = (hypothesis[0].to(device), hypothesis[1].to(device), hypothesis[2].to(device))
            labels = labels.to(device)
            pred = model(premise=premise, hypothesis=hypothesis)
            train_loss = criterion(pred, labels)
            train_loss.backward()
            optimizer.step()

            train_loss = train_loss.cpu().item()
            train_acc = accuracy(pred, labels).item()
            counter += 1
            
            if log:
                if counter % 10 == 0:
                    wandb.log({
                            'Training Loss': train_loss,
                            'Training Accuracy': train_acc, 
                        }, step=counter)
            
            if counter % report_period == 0:
                model.eval()
                lr_scheduler.step()
                with torch.no_grad():
                    val_acc_all = 0
                    val_loss_all = 0
                    val_total = 0
                    for premise, hypothesis, labels in test_loader:
                        premise = (premise[0].to(device), premise[1].to(device), premise[2].to(device))
                        hypothesis = (hypothesis[0].to(device), hypothesis[1].to(device), hypothesis[2].to(device))
                        labels = labels.to(device)
                        pred = model(premise=premise, hypothesis=hypothesis)
                        val_loss = criterion(pred, labels)
                        val_acc = accuracy(pred, labels)
                        val_acc_all += val_acc
                        val_loss_all += val_loss
                        val_total += 1

                        
                    val_acc = (val_acc_all / val_total).item()
                    val_loss = (val_loss_all / val_total).item()

                    print ('Epoch [{}/{}], Step [{}/{}], Training Loss: {:.4f}, Training acc: {:.4f}  Test Loss: {:.4f}   Test Acc: {:.4f}'.\
                           format(epoch+1, n_epochs, batch+1, train_batchs_per_epoch, train_loss, train_acc, val_loss, val_acc))

                    if log:
                        wandb.log({
                                'Validation Loss': val_loss,
                                'Validation Accuracy': val_acc, 
                            }, step=counter)

                    if val_acc > best_val_acc:
                        torch.save(model.state_dict(), f'./{data_name}_{context}_state_dict.pt')


def train_single_encoder(model, train_loader, test_loader, optimizer, 
                         n_epochs, data_name, context, device, criterion, lr_scheduler,
                         report_period=50, wandb=wandb, log=True):
    counter = 0
    best_val_acc = 0
    model = model.to(device)
    train_batchs_per_epoch = len(train_loader)
    for epoch in range(n_epochs):
        for batch ,(tokens, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            tokens = (tokens[0].to(device), tokens[1].to(device), tokens[2].to(device))
            labels = labels.to(device)
            pred = model(tokens)
            train_loss = criterion(pred, labels)
            train_loss.backward()
            optimizer.step()

            train_loss = train_loss.cpu().item()
            train_acc = accuracy(pred, labels).item()
            counter += 1
            
            if log:
                if counter % 10 == 0:
                    wandb.log({
                            'Training Loss': train_loss,
                            'Training Accuracy': train_acc, 
                        }, step=counter)
            
            if counter % report_period == 0:
                lr_scheduler.step()
                model.eval()
                with torch.no_grad():
                    val_acc_all = 0
                    val_loss_all = 0
                    val_total = 0
                    for tokens, labels in test_loader:
                        tokens = (tokens[0].to(device), tokens[1].to(device), tokens[2].to(device))
                        labels = labels.to(device)
                        pred = model(tokens)
                        val_loss = criterion(pred, labels)
                        val_acc = accuracy(pred, labels)
                        val_acc_all += val_acc
                        val_loss_all += val_loss
                        val_total += 1

                        
                    val_acc = (val_acc_all / val_total).item()
                    val_loss = (val_loss_all / val_total).item()

                    print ('Epoch [{}/{}], Step [{}/{}], Training Loss: {:.4f}, Training acc: {:.4f}  Test Loss: {:.4f}   Test Acc: {:.4f}'.\
                           format(epoch+1, n_epochs, batch+1, train_batchs_per_epoch, train_loss, train_acc, val_loss, val_acc))

                    if log:
                        wandb.log({
                                'Validation Loss': val_loss,
                                'Validation Accuracy': val_acc, 
                            }, step=counter)

                    if val_acc > best_val_acc:
                        torch.save(model.state_dict(), f'./{data_name}_{context}_state_dict.pt')
    
def train_adv_cls(model, train_loader, test_loader, optimizer, 
                  delta, n_epochs, data_name, context, device, criterion, lr_scheduler,
                  report_period=50, wandb=wandb, log=True):
    counter = 0
    best_val_acc = 0
    model = model.to(device)
    train_batchs_per_epoch = len(train_loader)
    for epoch in range(n_epochs):
        for batch ,(premise, hypothesis, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            premise = (premise[0].to(device), premise[1].to(device), premise[2].to(device))
            hypothesis = (hypothesis[0].to(device), hypothesis[1].to(device), hypothesis[2].to(device))
            labels = labels.to(device)
            label_prediction_combined, label_prediction_hypothesis = model(premise=premise, hypothesis=hypothesis)
            label_prediction_combined_loss = criterion(label_prediction_combined, labels)
            label_prediction_hypothesis_loss = criterion(label_prediction_hypothesis, labels)
            train_loss = label_prediction_combined_loss + delta * label_prediction_hypothesis_loss
            train_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                train_loss = train_loss.cpu().item()
                train_acc = accuracy(label_prediction_combined, labels).item()
                train_hypothesis_loss = label_prediction_hypothesis_loss.cpu().item()
                train_hypothesis_acc = accuracy(label_prediction_hypothesis, labels).item()
            
            counter += 1
            
            if log:
                if counter % 10 == 0:
                    wandb.log({
                            'Training Combined Loss': train_loss,
                            'Training Combined Accuracy': train_acc,
                            'Training Hypothesis Only Loss': train_hypothesis_loss,
                            "Training Hypothesis Only Accuracy": train_hypothesis_acc
                        }, step=counter)
            
            if counter % report_period == 0:
                lr_scheduler.step()
                model.eval()
                with torch.no_grad():
                    val_acc_all = 0
                    val_loss_all = 0
                    val_total = 0
                    test_hypothesis_acc_all = 0
                    test_hypothesis_loss_all = 0
                    for premise, hypothesis, labels in test_loader:
                        premise = (premise[0].to(device), premise[1].to(device), premise[2].to(device))
                        hypothesis = (hypothesis[0].to(device), hypothesis[1].to(device), hypothesis[2].to(device))
                        labels = labels.to(device)
                        label_prediction_combined, label_prediction_hypothesis = model(premise=premise, hypothesis=hypothesis)
                        label_prediction_combined_loss = criterion(label_prediction_combined, labels)
                        label_prediction_hypothesis_loss = criterion(label_prediction_hypothesis, labels)
                        val_loss = label_prediction_combined_loss + delta * label_prediction_hypothesis_loss
                        val_acc = accuracy(label_prediction_combined, labels)
                        test_hypothesis_loss = label_prediction_hypothesis_loss.cpu().item()
                        test_hypothesis_acc = accuracy(label_prediction_hypothesis, labels).item()
                        val_acc_all += val_acc
                        val_loss_all += val_loss
                        test_hypothesis_loss_all += test_hypothesis_loss 
                        test_hypothesis_acc_all += test_hypothesis_acc 
                        val_total += 1

                        
                    val_acc = (val_acc_all / val_total).item()
                    val_loss = (val_loss_all / val_total).item()
                    test_hypothesis_loss = (test_hypothesis_loss_all / val_total)
                    test_hypothesis_acc = (test_hypothesis_acc_all / val_total)
                    

                    print ('Epoch [{}/{}], Step [{}/{}], Training Loss: {:.4f}, Training Acc: {:.4f}  Test Loss: {:.4f}   Test Acc: {:.4f} Train Hypothesis Only Loss {:.4f} \
                    Train Hypothesis Only Acc {:.4f}   Test Hypothesis Only Loss {:.4f}   Test Hypothesis Only Acc {:.4f}'.\
                           format(epoch+1, n_epochs, batch+1, train_batchs_per_epoch, train_loss, train_acc, val_loss, val_acc, train_hypothesis_loss, train_hypothesis_acc,
                                  test_hypothesis_loss, test_hypothesis_acc))
                    

                    if log:
                        wandb.log({
                                'Validation Loss': val_loss,
                                'Validation Accuracy': val_acc, 
                                'Validation Hypothesis Only Loss': test_hypothesis_loss,
                                "Validation Hypothesis Only Accuracy": test_hypothesis_acc
                            }, step=counter)

                    if val_acc > best_val_acc:
                        torch.save(model.state_dict(), f'./{data_name}_{context}_state_dict.pt')
                        
def train_adv_dat(model, train_loader, test_loader, p, optimizer,
                  n_epochs, data_name, context, device, criterion, lr_scheduler,
                  report_period=50, wandb=wandb, log=True):
    counter = 0
    best_val_acc = 0
    model = model.to(device)
    train_batchs_per_epoch = len(train_loader)
    for epoch in range(n_epochs):
        for batch ,(premise, hypothesis, labels, perturb_mask) in enumerate(train_loader):
            train_unperturb_loss, train_perturb_loss = 0,0
            model.train()
            optimizer.zero_grad()
            premise = (premise[0].to(device), premise[1].to(device), premise[2].to(device))
            hypothesis = (hypothesis[0].to(device), hypothesis[1].to(device), hypothesis[2].to(device))
            labels = labels.to(device)
            perturb_mask = perturb_mask.to(device)
            unperturb_pred, perturb_pred = model(premise=premise, hypothesis=hypothesis, perturb_mask=perturb_mask)
            if unperturb_pred != None:
                train_unperturb_loss = criterion(unperturb_pred, labels[perturb_mask==0])
            if perturb_pred != None:
                train_perturb_loss = criterion(perturb_pred, labels[perturb_mask==1])
            train_loss = (1-p) * train_unperturb_loss + p * train_perturb_loss
            train_loss.backward()
            optimizer.step()
            train_loss = train_loss.item()
            with torch.no_grad():
                if unperturb_pred != None:
                    train_acc_unperturb = accuracy(unperturb_pred, labels[perturb_mask==0]).item()
                if perturb_pred != None:
                    train_acc_perturb = accuracy(perturb_pred, labels[perturb_mask==1]).item()
                
            
            counter += 1
            
            if log:
                if counter % 10 == 0:
                    wandb.log({
                            'train total loss': train_loss ,
                            'train loss unperturb': train_unperturb_loss,
                            'train accuracy unperturb': train_acc_unperturb,
                            'train loss perturb': train_perturb_loss,
                            "train accuracy perturb": train_acc_perturb
                        }, step=counter)
            
            if counter % report_period == 0:
                lr_scheduler.step()
                model.eval()
                with torch.no_grad():
                    val_loss_unperturb_all = 0
                    val_acc_unperturb_all = 0
                    val_loss_perturb_all = 0
                    val_acc_perturb_all = 0
                    val_loss_all = 0
                    val_total = 0
                    for premise, hypothesis, labels, perturb_mask in test_loader:
                        unperturb_loss, perturb_loss = 0, 0
                        premise = (premise[0].to(device), premise[1].to(device), premise[2].to(device))
                        hypothesis = (hypothesis[0].to(device), hypothesis[1].to(device), hypothesis[2].to(device))
                        labels = labels.to(device)
                        perturb_mask = perturb_mask.to(device)
                        unperturb_pred, perturb_pred = model(premise=premise, hypothesis=hypothesis, perturb_mask=perturb_mask)
                        if unperturb_pred != None:
                            unperturb_loss = criterion(unperturb_pred, labels[perturb_mask==0])
                            val_loss_unperturb = criterion(unperturb_pred, labels[perturb_mask==0]).item()
                            val_acc_unperturb = accuracy(unperturb_pred, labels[perturb_mask==0]).item()
                            val_loss_unperturb_all += val_loss_unperturb if not np.isnan(val_loss_unperturb) else 0
                            val_acc_unperturb_all += val_acc_unperturb if not np.isnan(val_acc_unperturb) else 0
                        if perturb_pred != None:
                            perturb_loss = criterion(perturb_pred, labels[perturb_mask==1])
                            val_loss_perturb = criterion(perturb_pred, labels[perturb_mask==1]).item()
                            val_acc_perturb = accuracy(perturb_pred, labels[perturb_mask==1]).item()
                            val_loss_perturb_all += val_loss_perturb if not np.isnan(val_loss_perturb) else 0
                            val_acc_perturb_all += val_acc_perturb if not np.isnan(val_acc_perturb) else 0
                            
                        val_loss = (1-p) * unperturb_loss + p * perturb_loss
                        val_loss = val_loss.item()
                        val_loss_all += val_loss
                        val_total += 1

                        
                    aver_loss_unperturb = val_loss_unperturb_all / val_total
                    aver_acc_unperturb = val_acc_unperturb_all / val_total
                    aver_loss_perturb = val_loss_perturb_all / val_total
                    aver_acc_perturb = val_acc_perturb_all / val_total
                    aver_loss = val_loss_all/val_total
                    

                    print ('Epoch [{}/{}], Step [{}/{}], Training Total Loss: {:.4f}, Training Loss Unperturb: {:.4f}  Training Loss Perturb: {:.4f}   \
                           Training Acc Unperturb: {:.4f} Training Acc Perturb {:.4f} \
                    Validation Total Loss: {:.4f}, Validation Loss Unperturb: {:.4f}  Validation Loss Perturb: {:.4f}   \
                           Validation Acc Unperturb: {:.4f} Validation Acc Perturb {:.4f} '.\
                           format(epoch+1, n_epochs, batch+1, train_batchs_per_epoch, train_loss, train_unperturb_loss, train_perturb_loss,
                                  train_acc_unperturb, train_acc_perturb, aver_loss, aver_loss_unperturb, aver_loss_perturb,
                                  aver_acc_unperturb, aver_acc_perturb))
                    

                    if log:
                        wandb.log({
                                'Validation Loss': aver_loss,
                                'Validation loss unperturb': aver_loss_unperturb, 
                                'Validation loss perturb': aver_loss_perturb,
                                "Validation acc unperturb": aver_acc_unperturb,
                                "Validation acc perturb": aver_acc_perturb
                            }, step=counter)

                    if aver_loss > best_val_acc:
                        torch.save(model.state_dict(), f'./{data_name}_{context}_state_dict.pt')
                        best_val_acc = aver_loss
def train_hypothesis_cls(model, train_loader, test_loader, optimizer, 
                  n_epochs, data_name, context, device, criterion, lr_scheduler,
                  report_period=50, wandb=wandb, log=True):
    counter = 0
    best_val_acc = 0
    model = model.to(device)
    train_batchs_per_epoch = len(train_loader)
    for epoch in range(n_epochs):
        for batch ,(premise, hypothesis, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            premise = (premise[0].to(device), premise[1].to(device), premise[2].to(device))
            hypothesis = (hypothesis[0].to(device), hypothesis[1].to(device), hypothesis[2].to(device))
            labels = labels.to(device)
            label_prediction_hypothesis = model(hypothesis)
            train_loss = criterion(label_prediction_hypothesis, labels)
            train_loss.backward()
            optimizer.step()

            train_loss = train_loss.cpu().item()
            train_acc = accuracy(label_prediction_hypothesis, labels).item()
            
            counter += 1
            
            if log:
                if counter % 10 == 0:
                    wandb.log({
                            'Training Loss': train_loss,
                            'Training Accuracy': train_acc, 
                        }, step=counter)
            
            if counter % report_period == 0:
                lr_scheduler.step()
                model.eval()
                with torch.no_grad():
                    val_acc_all = 0
                    val_loss_all = 0
                    val_total = 0
                    for premise, hypothesis, labels in test_loader:
                        premise = (premise[0].to(device), premise[1].to(device), premise[2].to(device))
                        hypothesis = (hypothesis[0].to(device), hypothesis[1].to(device), hypothesis[2].to(device))
                        labels = labels.to(device)
                        label_prediction_hypothesis = model(hypothesis)
                        val_loss = criterion(label_prediction_hypothesis, labels)
                        val_acc = accuracy(label_prediction_hypothesis, labels)
                        val_acc_all += val_acc
                        val_loss_all += val_loss
                        val_total += 1

                        
                    val_acc = (val_acc_all / val_total).item()
                    val_loss = (val_loss_all / val_total).item()
                    

                    print ('Epoch [{}/{}], Step [{}/{}], Training Loss: {:.4f}, Training Acc: {:.4f}  Test Loss: {:.4f}   Test Acc: {:.4f}'.\
                           format(epoch+1, n_epochs, batch+1, train_batchs_per_epoch, train_loss, train_acc, val_loss, val_acc))
                    

                    if log:
                        wandb.log({
                                'Validation Loss': val_loss,
                                'Validation Accuracy': val_acc, 
                            }, step=counter)

                    if val_acc > best_val_acc:
                        torch.save(model.state_dict(), f'./{data_name}_{context}_state_dict.pt')