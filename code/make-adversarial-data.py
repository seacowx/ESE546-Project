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

    anli_data = test_data + val_data

    counter = len(anli_data)
    pbar = tqdm(total=(args.size - counter))
    for idx in train_idx:
        anli_data.append(train_data[int(idx)])
        counter += 1
        if counter == args.size:
            break
        pbar.update(1)
    
    with open ('../anli_data.pkl', 'wb') as f:
        pickle.dump(anli_data, f)
    f.close()

if __name__ == '__main__':
    main()
