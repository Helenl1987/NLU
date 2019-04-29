import os

import torch
from torch.utils.data import Dataset

from options import opt

import pandas as pd
import logging
# import nltk
# from nltk.tokenize import WordPunctTokenizer
# from nltk.tokenize import word_tokenize
from spacy.lang.en import English
from spacy.lang.fr import French
from spacy.lang.de import German


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

    def check_unk_tok(self):
        total_tok = 0
        total_unk = 0
        for x in self.X:
            total_tok += len(x['inputs'])
            counter = pd.value_counts(x['inputs'])
            if 0 in counter.keys():
                total_unk += counter[0]
                # logging.warning('\n%s\n' % ' '.join(x['tokens']))
                # logging.warning('\n%s\n' % (' $ '.join('%s:%d' % (x['tokens'][idx], x['inputs'][idx]) for idx in range(len(x['tokens'])))))
                # logging.warning('%s\n%s' % (' '.join(x['tokens']), ' '.join(str(e) for e in x['inputs'])))
        return total_tok, total_unk

def clean_sentence(sent):
    sent = sent.replace('<br />', ' ').replace('<br>', ' ').replace('<BR>', ' ').replace('\\', '').replace('&quot', ' ')
    return sent

def read_mtl_file(domain, filename):
    X = []
    Y = []
    if domain == 'en':
        # tokenizer = WordPunctTokenizer()
        tokenizer = English().Defaults.create_tokenizer()
    elif domain == 'fr':
        # tokenizer = nltk.data.load('tokenizers/punkt/french.pickle')
        tokenizer = French().Defaults.create_tokenizer()
    elif domain == 'de':
        # tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')
        tokenizer = German().Defaults.create_tokenizer()
    with open(filename, 'r', encoding='utf-8') as inf:
        for line in inf.readlines():
            parts = line.split('\t')
            if len(parts) == 3: # labeled
                Y.append(int(float(parts[1])))
            elif len(parts) == 2: # unlabeled
                Y.append(0)
            else:
                raise Exception('Unknown format')
            clean = clean_sentence(parts[-1])
            # if domain is 'en':
            #     words = word_tokenize(clean, language='english')
            # elif domain is 'fr':
            #     words = word_tokenize(clean, language='french')
            # elif domain is 'de':
            #     words = word_tokenize(clean, language='german')
            words = [str(e) for e in tokenizer(clean)]
            X.append({'tokens': words})
    Y = torch.LongTensor(Y).to(opt.device)
    return (X, Y)


def get_fdu_mtl_datasets(vocab, data_dir, domain, max_seq_len):
    print('Loading FDU MTL data for {} Domain'.format(domain))
    # train and dev set
    train_X, train_Y = read_mtl_file(domain, os.path.join(data_dir, '{}_{}'.format(domain, opt.topic_domain)))

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
    unlabeled_X, unlabeled_Y = read_mtl_file(domain, os.path.join(data_dir, '{}_{}_unlabel'.format(domain, opt.topic_domain)))

    unlabeled_set = FduMtlDataset(unlabeled_X, unlabeled_Y, max_seq_len)
    vocab.prepare_inputs(unlabeled_set, domain)

    #check number of unknown tokens
    # if domain is 'fr':
    # p = train_set.check_unk_tok()
    # print('unk for train_set: %d, %d' % (p[0], p[1]))
    # p = test_set.check_unk_tok()
    # print('unk for test_set: %d, %d' % (p[0], p[1]))
    # p = dev_set.check_unk_tok()
    # print('unk for dev_set: %d, %d' % (p[0], p[1]))
    # p = unlabeled_set.check_unk_tok()
    # print('unk for unlabeled_set: %d, %d' % (p[0], p[1]))

    return train_set, dev_set, test_set, unlabeled_set



