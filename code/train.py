import time
import json
import utils
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import autoaug

import torch
import wandb
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import BertForSequenceClassification, AdamW, BertTokenizer


def train(model, train_loader, validation_loader, n_epochs, lr_rate, data_name, content, log=False):
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=True)
    # optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad], lr=lr_rate)
    if log:
        config = dict(
            learning_rate=lr_rate,
            architecture="BERT",
            dataset_id="HW4-P3-MLP-546",
        )

        wandb.init(
            project=f"ESE546-Project-{data_name.capitalize()}",
            config=config
        )

    counter = 0
    best_val_acc = 0
    total_train_loss = 0
    for e in tqdm(range(n_epochs), position=0, leave=False):
        for input_ids, token_type_ids, attention_mask, lab in tqdm(train_loader, position=1, leave=False):
            optimizer.zero_grad()
            input_ids, token_type_ids, attention_mask, lab = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device), lab.to(device)
            loss, pred = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=lab).values()
            total_train_loss += loss.cpu().item()

            pred = torch.argmax(pred, dim=1)
            train_acc = accuracy_score(lab.cpu(), pred.cpu())

            loss.backward()
            optimizer.step()
            counter += 1
            if log:
                if counter % 10 == 0:
                    wandb.log({
                            'Training Loss': total_train_loss / counter,
                            'Training Accuracy': train_acc, 
                        })
            
            if counter % 500 == 0:
                with torch.no_grad():
                    val_acc_all = 0
                    val_loss_all = 0
                    val_total = 0
                    for input_ids, token_type_ids, attention_mask, val_lab in tqdm(validation_loader, position=1, leave=False): 
                        input_ids, token_type_ids, attention_mask, val_lab = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device), val_lab.to(device)
                        val_loss, val_pred = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=val_lab).values()
                        val_pred = torch.argmax(val_pred, dim=1)
                        val_acc = accuracy_score(val_lab.cpu(), val_pred.cpu())
                        val_acc_all += val_acc
                        val_loss_all += val_loss
                        val_total += 1

                    val_acc = val_acc_all / val_total
                    val_loss = val_loss_all / val_total

                    if log:
                        wandb.log({
                                'Validation Loss': val_loss,
                                'Validation Accuracy': val_acc, 
                            })

                    if val_acc > best_val_acc:
                        torch.save(model.state_dict(), f'../state-dicts/{data_name}_{content}_state_dict.pt')


parser = argparse.ArgumentParser()
parser.add_argument(
    '--train_size', type=int, default=50000, help='size of training dataset'
)
parser.add_argument(
    '--val_size', type=int, default=10000, help='size of validation dataset'
)
parser.add_argument(
    '--batch_size', type=int, default=16, help='batch size of the network'
)
parser.add_argument(
    '--n_epochs', type=int, default=10, help='number of epochs'
)
parser.add_argument(
    '--lr_rate', type=float, default=2e-4, help='learning rate'
)
parser.add_argument(
    '--dataset', type=str, default='snli', help='dataset used for training. pick from "snli", "hans", "multi_nli", "anli_diy"'
)
parser.add_argument(
    '--content', type=str, default='all', help='choose from "all" for premise + hypothesis; "premise" for premise-only and "hypothesis" for hypothesis only'
)
parser.add_argument(
    '--apply_augment', type=bool, default=False, help='apply data augmentation to train data or not'
)


def main():
    args = parser.parse_args()

    if args.dataset != 'anli_diy':
        metadata, _ = utils.load_data(args.dataset)
    else: 
        with open('../anli_diy.json', 'r') as f:
            train_data = json.load(f)
        f.close()
        metadata, _ = utils.load_data('snli')
        _, val_data, _ = metadata.values()

    if args.dataset == 'snli':
        _, val_data, train_data = metadata.values()
    elif args.dataset == 'hans':
        train_data, val_data = metadata.values()
    elif args.dataset == 'multi_nli':
        train_data, val_data, _ = metadata.values()

    train_data = train_data[:args.train_size]
    val_data = val_data[:args.val_size]

    if args.apply_augment:
        train_data = autoaug.augment_data(train_data) #original train_size: 550k
        val_data = autoaug.transform_label(val_data) #original val/test size: 10k
    
    train_data = utils.NLIDataset(train_data, content=args.content)
    validation_data = utils.NLIDataset(val_data, content=args.content)

    train_loader = train_data.get_data_loaders(args.batch_size)
    validation_loader = validation_data.get_data_loaders(args.batch_size)

    # # * Functions for sanity check
    # train_ids, _, _, labels = next(iter(train_loader))
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # train_tokens = tokenizer.batch_decode(train_ids, skip_special_tokens=True)
    # for t, l in zip(train_tokens, labels):
    #     print(t)
    #     print(l)
    #     print('\n')
    # raise SystemExit()
    C = 2 if args.apply_augment else 3
    nli_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=C)
    nli_model.to(device)
    train(nli_model, train_loader, validation_loader, args.n_epochs, args.lr_rate, args.dataset, args.content, log=True)


if __name__ == '__main__':
    utils.set_seed()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    main()
    



