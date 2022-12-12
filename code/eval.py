import utils
import autoaug
import random
import argparse
from tqdm import tqdm
from datasets import concatenate_datasets
from transformers import BertForSequenceClassification, BertTokenizer

import torch
from sklearn.metrics import accuracy_score, f1_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def eval(model, dataloader, binary):
    overall_acc = 0
    overall_loss = 0
    overall_f1 = 0
    for idx, (input_ids, token_type_ids, attention_mask, lab) in enumerate(tqdm(dataloader, position=0, leave=False)):
        input_ids, token_type_ids, attention_mask, lab = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device), lab.to(device)
        loss, pred = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=lab).values()

        if binary:
            pred = torch.argmax(pred, dim=1)
            pred = [l if l == 0 else 1 for l in pred.cpu()]
            overall_acc += accuracy_score(lab.cpu(), pred)
            overall_f1 += f1_score(lab.cpu(), pred, average='macro')
        else:
            pred = torch.argmax(pred, dim=1)
            overall_acc += accuracy_score(lab.cpu(), pred.cpu())
            overall_f1 += f1_score(lab.cpu(), pred.cpu(), average='macro')

        overall_loss += loss.cpu().item()
    return overall_acc / (idx + 1), overall_f1 / (idx + 1), overall_loss / (idx + 1)
    

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size', type=int, default=16, help='batch size of the network'
)
parser.add_argument(
    '--test_size', type=int, default=10000, help='size of test dataset'
)
parser.add_argument(
    '--after_augment', type=bool, default=False, help='whether train data is augmented or not (if so, 2 classes for SNLI)'
)
parser.add_argument(
    '--eval_dataset', type=str, default='snli', help='choose from ["snli", "hans", "anli"]'
)
parser.add_argument(
    '--state_dict', type=str, help='name of the state dict'
)


def main():
    args = parser.parse_args()

    C = 2 if args.after_augment else 3

    nli_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=C).to(device)
    nli_model.load_state_dict(torch.load(f'../state-dicts/{args.state_dict}.pt'))

    if args.eval_dataset == 'snli':
        metadata, _ = utils.load_data('snli')
        test_data, _, _ = metadata.values()

        idx = random.sample(range(len(test_data)), args.test_size)
        test_data = test_data[:args.test_size]  

        if args.after_augment:
            test_data = autoaug.transform_label(test_data)
            print('Random guess acc:', sum([1 if lab == 0 or lab == 1 else 0 for lab in test_data['label']])/len(test_data['label']))
        binary = False

    
    elif args.eval_dataset == 'hans':
        metadata, _ = utils.load_data('hans')
        _, val_data = metadata.values()

        idx = random.sample(range(len(val_data)), args.test_size)
        test_data = val_data[idx]

        print('Random guess acc:', sum(val_data['label'])/len(val_data['label']))
        binary = True
    
    elif args.eval_dataset == 'anli':
        metadata, _ = utils.load_data('anli')
        _, _, test_r1, _, _, test_r2, _, _, test_r3 = metadata.values()
        test_data = concatenate_datasets([test_r1, test_r2, test_r3])
        binary = False
    
    elif args.eval_dataset == 'mnli':
        metadata, _ = utils.load_data('multi_nli')
        _, _, test_data = metadata.values()
        binary = False

    test_data = utils.NLIDataset(test_data)
    test_loader = test_data.get_data_loaders(args.batch_size)

    acc, f1, _ = eval(nli_model, test_loader, binary)
    print(f'Accuracy score: {acc}')
    print(f'F1 score: {f1}')


    # # * Functions for sanity check
    # train_ids, _, _, labels = next(iter(train_loader))
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # train_tokens = tokenizer.batch_decode(train_ids, skip_special_tokens=True)
    # for t, l in zip(train_tokens, labels):
    #     print(t)
    #     print(l)
    #     print('\n')
    # raise SystemExit()




if __name__ == '__main__':
    utils.set_seed()
    main()