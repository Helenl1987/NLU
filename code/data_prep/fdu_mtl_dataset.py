import os
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler
from options import opt
import pandas as pd
import logging
# import nltk
# from nltk.tokenize import WordPunctTokenizer
# from nltk.tokenize import word_tokenize
from spacy.lang.en import English
from spacy.lang.fr import French
from spacy.lang.de import German
from pytorch_pretrained_bert import BertTokenizer

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, emb_ids, input_ids, input_mask, segment_ids, label_id):
        self.emb_ids = emb_ids
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

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
            tmp = {}
            tmp['tokens'] = words
            tmp['sent'] = clean
            X.append(tmp)
    #Y = torch.LongTensor(Y).to(opt.device)
    return (X, Y)

def convert_to_features(X, Y, max_seq_length, bert_tokenizer):
    features = []
    for exi, example in enumerate(X):
        tokens_a = bert_tokenizer.tokenize(example['sent'])
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        if len(example['inputs']) < max_seq_length:
            tmp = [0]* max_seq_length
            tmp[:len(example['inputs'])] = example['inputs']
            example['inputs'] = tmp[:]
        elif len(example['inputs']) > max_seq_length:
            example['inputs'] = example['inputs'][:max_seq_length]

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        features.append(
            InputFeatures(emb_ids=example['inputs'],
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=Y[exi]))

    return features

def get_fdu_mtl_datasets(vocab, data_dir, domain, max_seq_len):
    print('Loading FDU MTL data for {} Domain'.format(domain))
    # train and dev set
    train_X, train_Y = read_mtl_file(domain, os.path.join(data_dir, '{}_{}'.format(domain, opt.topic_domain)))

    unit = len(train_X) // 10
    dev_X, dev_Y = train_X[-unit:], train_Y[-unit:]
    test_X, test_Y = train_X[-3*unit:-unit], train_Y[-3*unit:-unit]
    train_X, train_Y = train_X[:-3*unit], train_Y[:-3*unit]
    #train_set = FduMtlDataset(train_X, train_Y, max_seq_len)
    #dev_set = FduMtlDataset(dev_X, dev_Y, max_seq_len)
    #test_set = FduMtlDataset(test_X, test_Y, max_seq_len)
    # pre-compute embedding indices
    vocab.prepare_inputs(train_X, domain)
    vocab.prepare_inputs(dev_X, domain)
    vocab.prepare_inputs(test_X, domain)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    train_features = convert_to_features(train_X, train_Y, max_seq_len, bert_tokenizer)
    dev_features = convert_to_features(dev_X, dev_Y, max_seq_len, bert_tokenizer)
    test_features = convert_to_features(test_X, test_Y, max_seq_len, bert_tokenizer)

    all_train_emb_ids = torch.tensor([f.emb_ids for f in train_features], dtype=torch.long)
    all_train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_train_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_train_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_train_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_train_emb_ids, all_train_input_ids, all_train_input_mask, all_train_segment_ids, all_train_label_ids)


    all_test_emb_ids = torch.tensor([f.emb_ids for f in test_features], dtype=torch.long)
    all_test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_test_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_test_emb_ids, all_test_input_ids, all_test_input_mask, all_test_segment_ids, all_test_label_ids)

    all_dev_emb_ids = torch.tensor([f.emb_ids for f in dev_features], dtype=torch.long)
    all_dev_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
    all_dev_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
    all_dev_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
    all_dev_label_ids = torch.tensor([f.label_id for f in dev_features], dtype=torch.long)
    dev_data = TensorDataset(all_dev_emb_ids, all_dev_input_ids, all_dev_input_mask, all_dev_segment_ids,all_dev_label_ids)

    # unlabeled set
    unlabeled_X, unlabeled_Y = read_mtl_file(domain, os.path.join(data_dir, '{}_{}_unlabel'.format(domain, opt.topic_domain)))

    #unlabeled_set = FduMtlDataset(unlabeled_X, unlabeled_Y, max_seq_len)
    vocab.prepare_inputs(unlabeled_X, domain)
    unlabeled_features = convert_to_features(unlabeled_X, unlabeled_Y, max_seq_len, bert_tokenizer)
    all_unlabeled_emb_ids = torch.tensor([f.emb_ids for f in unlabeled_features], dtype=torch.long)
    all_unlabeled_input_ids = torch.tensor([f.input_ids for f in unlabeled_features], dtype=torch.long)
    all_unlabeled_input_mask = torch.tensor([f.input_mask for f in unlabeled_features], dtype=torch.long)
    all_unlabeled_segment_ids = torch.tensor([f.segment_ids for f in unlabeled_features], dtype=torch.long)
    all_unlabeled_label_ids = torch.tensor([f.label_id for f in unlabeled_features], dtype=torch.long)
    unlabeled_data = TensorDataset(all_unlabeled_emb_ids, all_unlabeled_input_ids, all_unlabeled_input_mask,
                                   all_unlabeled_segment_ids, all_unlabeled_label_ids)
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

    return train_data, dev_data, test_data, unlabeled_data



