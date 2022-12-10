import utils
import autoaug
import random
import argparse
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

import torch
from sklearn.metrics import accuracy_score, f1_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def eval(model, dataloader, snli):
    overall_acc = 0
    overall_loss = 0
    overall_f1 = 0
    for idx, (input_ids, token_type_ids, attention_mask, lab) in enumerate(tqdm(dataloader, position=0, leave=False)):
        input_ids, token_type_ids, attention_mask, lab = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device), lab.to(device)
        loss, pred = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=lab).values()

        if snli:
            pred = torch.argmax(pred, dim=1)
            overall_acc += accuracy_score(lab.cpu(), pred.cpu())
            overall_f1 += f1_score(lab.cpu(), pred.cpu(), average='macro')
        else:
            pred = torch.argmax(pred, dim=1)
            pred = [l if l == 0 else 1 for l in pred.cpu()]
            overall_acc += accuracy_score(lab.cpu(), pred)
            overall_f1 += f1_score(lab.cpu(), pred, average='macro')

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
    '--eval_snli', type=int, default=0, help='whether to evaluate on snli'
)
parser.add_argument(
    '--state_dict', type=str, help='name of the state dict'
)


def main():
    args = parser.parse_args()

    C = 2 if args.after_augment else 3

    nli_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=C).to(device)
    nli_model.load_state_dict(torch.load(f'./{args.state_dict}.pt'))

    if args.eval_snli:
        metadata, templates = utils.load_data('snli')
        test_data, _, _ = metadata.values()

        idx = random.sample(range(len(test_data)), args.test_size)
        test_data = test_data[:args.test_size]

        print('Random guess acc:', sum([1 if lab == 0 or lab == 1 else 0 for lab in test_data['label']])/len(test_data['label']))

        if args.after_augment:
            test_data = autoaug.transform_label(test_data)

        test_data = utils.NLIDataset(test_data)
        test_loader = test_data.get_data_loaders(args.batch_size)

        acc, f1, _ = eval(nli_model, test_loader, args.eval_snli)
        print(f'Accuracy score: {acc}')
        print(f'F1 score: {f1}')
    
    else:
        metadata, _ = utils.load_data('hans')
        train_data, val_data = metadata.values()

        idx = random.sample(range(len(val_data)), args.test_size)
        val_data = val_data[idx]

        print('Random guess acc:', sum(val_data['label'])/len(val_data['label']))

        train_data = utils.NLIDataset(train_data)
        val_data = utils.NLIDataset(val_data)

        val_loader = val_data.get_data_loaders(args.batch_size)

        acc, f1, _ = eval(nli_model, val_loader, args.eval_snli)
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