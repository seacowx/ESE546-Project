import pickle
import utils
import argparse
import numpy as np
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

import torch
from sklearn.metrics import accuracy_score, f1_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def eval(model, dataloader):
    overall_acc = 0
    overall_loss = 0
    overall_f1 = 0
    sample_idx_lst = []
    for idx, (input_ids, token_type_ids, attention_mask, lab) in enumerate(tqdm(dataloader, position=0, leave=False)):
        input_ids, token_type_ids, attention_mask, lab = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device), lab.to(device)
        batch_size = input_ids.size(0)
        loss, pred = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=lab).values()

        pred = torch.argmax(pred, dim=1)
        wrong_idx = (pred.cpu() != lab.cpu()).nonzero().numpy()
        wrong_idx = [lst for sublst in wrong_idx for lst in sublst] 
        wrong_idx = [i + idx * batch_size for i in wrong_idx]
        sample_idx_lst += wrong_idx

        overall_acc += accuracy_score(lab.cpu(), pred.cpu())
        overall_f1 += f1_score(lab.cpu(), pred.cpu(), average='macro')
        overall_loss += loss.cpu().item()

    print(overall_acc / (idx + 1), overall_f1 / (idx + 1))
    return sample_idx_lst
    

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size', type=int, default=16, help='batch size of the network'
)
parser.add_argument(
    '--state_dict', type=str, help='name of the state dict'
)


def main():
    args = parser.parse_args()

    nli_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3).to(device)
    nli_model.load_state_dict(torch.load(f'../state-dicts/{args.state_dict}.pt'))

    metadata, templates = utils.load_data('snli')
    test_data, val_data, train_data = metadata.values()

    test_data = utils.NLIDataset(test_data)
    val_data = utils.NLIDataset(val_data)
    train_data = utils.NLIDataset(train_data, size=np.inf)
    test_loader = test_data.get_data_loaders(args.batch_size, shuffle=False)
    val_loader = val_data.get_data_loaders(args.batch_size, shuffle=False)
    train_loader = train_data.get_data_loaders(args.batch_size, shuffle=False)

    idx_dict = {}
    test_idx = eval(nli_model, test_loader)
    val_idx = eval(nli_model, val_loader)
    train_idx = eval(nli_model, train_loader)

    print(idx_dict)

    with open('../adversarial-idx/test.pkl', 'wb') as f:
        pickle.dump(test_idx, f)
    f.close()

    with open('../adversarial-idx/val.pkl', 'wb') as f:
        pickle.dump(val_idx, f)
    f.close()

    with open('../adversarial-idx/train.pkl', 'wb') as f:
        pickle.dump(train_idx, f)
    f.close()


if __name__ == '__main__':
    utils.set_seed()
    main()