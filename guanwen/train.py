# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 11:50:54 2022

@author: nayut
"""
from utils import accuracy
import torch
def train_double_encoder(model, train_loader, test_loader, optimizer, 
                         n_epochs, data_name, context, device, criterion, lr_scheduler,
                         report_period=50):
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
            
            # if log:
            #     if counter % 10 == 0:
            #         wandb.log({
            #                 'Training Loss': training_loss,
            #                 'Training Accuracy': train_acc, 
            #             })
            
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

                    # if log:
                    #     wandb.log({
                    #             'Validation Loss': val_loss,
                    #             'Validation Accuracy': val_acc, 
                    #         })

                    if val_acc > best_val_acc:
                        torch.save(model.state_dict(), f'./{data_name}_{context}_state_dict.pt')


def train_single_encoder(model, train_loader, test_loader, optimizer, 
                         n_epochs, data_name, context, device, criterion, lr_scheduler,
                         report_period=50):
    counter = 0
    best_val_acc = 0
    model = model.to(device)
    train_batchs_per_epoch = len(train_loader)
    for epoch in range(n_epochs):
        for batch ,(tokens, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            tokens = (tokens[0].to(device), tokens[2].to(device))
            labels = labels.to(device)
            pred = model(tokens)
            train_loss = criterion(pred, labels)
            train_loss.backward()
            optimizer.step()

            train_loss = train_loss.cpu().item()
            train_acc = accuracy(pred, labels).item()
            counter += 1
            
            # if log:
            #     if counter % 10 == 0:
            #         wandb.log({
            #                 'Training Loss': training_loss,
            #                 'Training Accuracy': train_acc, 
            #             })
            
            if counter % report_period == 0:
                lr_scheduler.step()
                model.eval()
                with torch.no_grad():
                    val_acc_all = 0
                    val_loss_all = 0
                    val_total = 0
                    for tokens, labels in test_loader:
                        tokens = (tokens[0].to(device), tokens[2].to(device))
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

                    # if log:
                    #     wandb.log({
                    #             'Validation Loss': val_loss,
                    #             'Validation Accuracy': val_acc, 
                    #         })

                    if val_acc > best_val_acc:
                        torch.save(model.state_dict(), f'./{data_name}_{context}_state_dict.pt')
    
def train_adv_cls(model, train_loader, test_loader, optimizer, 
                  delta, n_epochs, data_name, context, device, criterion, lr_scheduler,
                  report_period=50):
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

            train_loss = train_loss.cpu().item()
            train_acc = accuracy(label_prediction_combined, labels).item()
            train_hypothesis_loss = label_prediction_hypothesis_loss.cpu().item()
            train_hypothesis_acc = accuracy(label_prediction_hypothesis, labels).item()
            
            counter += 1
            
            # if log:
            #     if counter % 10 == 0:
            #         wandb.log({
            #                 'Training Loss': training_loss,
            #                 'Training Accuracy': train_acc, 
            #             })
            
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
                    

                    # if log:
                    #     wandb.log({
                    #             'Validation Loss': val_loss,
                    #             'Validation Accuracy': val_acc, 
                    #         })

                    if val_acc > best_val_acc:
                        torch.save(model.state_dict(), f'./{data_name}_{context}_state_dict.pt')