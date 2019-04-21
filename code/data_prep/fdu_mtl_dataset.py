import os

import torch
from torch.utils.data import Dataset

from options import opt

class FduMtlDataset(Dataset):
    num_labels = 2
    def __init__(self, X, Y, max_seq_len):
        self.X = X
        self.Y = Y
        self.num_labels = 2
        if max_seq_len > 0:
            self.set_max_seq_len(max_seq_len)
        assert len(self.X) == len(self.Y), 'X and Y have different lengths'

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

    def set_max_seq_len(self, max_seq_len):
        for x in self.X:
            x['tokens'] = x['tokens'][:max_seq_len]
        self.max_seq_len = max_seq_len

    def get_max_seq_len(self):
        if not hasattr(self, 'max_seq_len'):
            self.max_seq_len = max([len(x) for x in self.X])
        return self.max_seq_len

def clean_sentence(sent):
    sent = sent.replace('<br />', ' ').replace('\\', '').replace('&quot', ' ')
    return sent

def read_mtl_file(filename):
    X = []
    Y = []
    with open(filename, 'r', encoding='ISO-8859-2') as inf:
        for line in inf.readlines():
            parts = line.split('\t')
            if len(parts) == 3: # labeled
                Y.append(int(float(parts[1])))
            elif len(parts) == 2: # unlabeled
                Y.append(0)
            else:
                raise Exception('Unknown format')
            clean = clean_sentence(parts[-1])
            words = clean.split(' ')
            X.append({'tokens': words})
    Y = torch.LongTensor(Y).to(opt.device)
    return (X, Y)


def get_fdu_mtl_datasets(vocab, data_dir, domain, max_seq_len):
    print('Loading FDU MTL data for {} Domain'.format(domain))
    # train and dev set
    train_X, train_Y = read_mtl_file(os.path.join(data_dir, '{}_book'.format(domain)))
    unit = len(train_X) // 10
    dev_X, dev_Y = train_X[-unit:], train_Y[-unit:]
    test_X, test_Y = train_X[-3*unit:-unit], train_Y[-3*unit:-unit]
    train_X, train_Y = train_X[:-3*unit], train_Y[:-3*unit]
    train_set = FduMtlDataset(train_X, train_Y, max_seq_len)
    dev_set = FduMtlDataset(dev_X, dev_Y, max_seq_len)
    test_set = FduMtlDataset(test_X, test_Y, max_seq_len)
    # pre-compute embedding indices
    vocab.prepare_inputs(train_set, domain)
    vocab.prepare_inputs(dev_set, domain)
    vocab.prepare_inputs(test_set, domain)


    # unlabeled set
    unlabeled_X, unlabeled_Y = read_mtl_file(os.path.join(data_dir, '{}_book_unlabel'.format(domain)))
    unlabeled_set = FduMtlDataset(unlabeled_X, unlabeled_Y, max_seq_len)
    vocab.prepare_inputs(unlabeled_set, domain)

    return train_set, dev_set, test_set, unlabeled_set
