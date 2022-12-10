import json
import utils
import pickle
import argparse
from tqdm import tqdm
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument(
    '--size', type=int, default=50000, help='default data size'
)

def main():
    args = parser.parse_args()

    idx_file_lst = glob('../adversarial-idx/*.pkl')
    for path in idx_file_lst:
        name = path.split('/')[-1].split('.')[0] + '_idx'
        with open(path, 'rb') as f:
            globals()[name] = pickle.load(f)

    metadata, templates = utils.load_data('snli')
    test_data, val_data, train_data = metadata.values()

    test_data = [test_data[int(i)] for i in test_idx]
    val_data = [val_data[int(i)] for i in val_idx]

    assert len(test_data) == len(test_idx)
    assert len(val_data) == len(val_idx)

    anli_data = {'premise': [], 'hypothesis': [], 'label': []}
    for data in test_data:
        anli_data['premise'].append(data['premise'])
        anli_data['hypothesis'].append(data['hypothesis'])
        anli_data['label'].append(data['label'])

    for data in val_data:
        anli_data['premise'].append(data['premise'])
        anli_data['hypothesis'].append(data['hypothesis'])
        anli_data['label'].append(data['label'])

    counter = len(anli_data['premise'])
    pbar = tqdm(total=(args.size - counter))
    for idx in train_idx:
        cur_data = train_data[int(idx)]
        anli_data['premise'].append(cur_data['premise'])
        anli_data['hypothesis'].append(cur_data['hypothesis'])
        anli_data['label'].append(cur_data['label'])
        counter += 1
        if counter == args.size:
            break
        pbar.update(1)
    
    with open ('../anli_diy.json', 'w') as f:
        json.dump(anli_data, f)
    f.close()

if __name__ == '__main__':
    main()
