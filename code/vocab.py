import numpy as np
import torch
import torch.nn as nn

from options import opt


class Vocab:
    def __init__(self, txt_dir):
        self.language_list = ['en', 'de', 'fr']
        self.language_size = len(self.language_list)
        self.vocab_size = 0
        self.unk_tok = '<unk>'
        self.unk_idx = 0
        self.eos_tok = '</s>'
        self.eos_idx = 1
        self.v2wvocab = [[] for i in range(len(self.language_list))]
        self.w2vvocab = [{} for i in range(len(self.language_list))]
        flag = True
        for lan_idx, language in enumerate(self.language_list):
            txt_file = txt_dir + 'wiki.multi.' + language + '.vec'
            with open(txt_file, 'r') as inf:
                parts = inf.readline().split()
                assert len(parts) == 2
                if flag:
                    local_vocab_size = int(parts[0])
                    local_emb_size  = int(parts[1])
                    self.vocab_size = local_vocab_size * len(self.language_list) + 2
                    self.emb_size = local_emb_size
                    self.embeddings = np.empty((self.vocab_size, self.emb_size), dtype=np.float)
                    cnt = 2
                    flag = False
                else:
                    assert int(parts[0]) == local_vocab_size
                    assert int(parts[1]) == local_emb_size
                # add an UNK token
                # self.v2wvocab[lan_idx] = ['<unk>']
                # self.w2vvocab[lan_idx] = {'<unk>': 0}
                for line in inf.readlines():
                    parts = line.rstrip().split(' ')
                    word = parts[0]
                    # add to vocab
                    self.v2wvocab[lan_idx].append(word)
                    self.w2vvocab[lan_idx][word] = cnt
                    # load vector
                    vector = [float(x) for x in parts[-self.emb_size:]]
                    self.embeddings[cnt] = vector
                    cnt += 1
            print('%s processed' % txt_file)
            # print('cnt = %d' % cnt)

        # opt.eos_idx = self.eos_idx = self.w2vvocab[lan_idx][self.eos_tok]
        opt.eos_idx = self.eos_idx
        # randomly initialize <unk> vector
        self.embeddings[self.unk_idx] = np.random.normal(0, 1, size=self.emb_size)
        self.embeddings[self.unk_idx] /= np.sum(self.embeddings[self.unk_idx])
        # normalize
        # self.embeddings /= np.linalg.norm(self.embeddings, axis=1).reshape(-1,1)
        # zero </s>
        self.embeddings[self.eos_idx] = 0

        opt.vocab_size = self.vocab_size
        opt.emb_size = self.emb_size

    def base_form(word):
        return word.strip().lower()

    def init_embed_layer(self):
        word_emb = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=self.eos_idx)
        if not opt.random_emb:
            word_emb.weight.data = torch.from_numpy(self.embeddings).float()
        return word_emb

    def lookup(self, word, language):
        lan_idx = self.language_list.index(language)
        word = Vocab.base_form(word)
        if word in self.w2vvocab[lan_idx]:
            return self.w2vvocab[lan_idx][word]
        return self.unk_idx

    def prepare_inputs(self, dataset, language):
        for x in dataset:
            x['inputs'] = [self.lookup(w, language) for w in x['tokens']]


# def main():
#     # emb_filename = '../data/MWE/'
#     # vocab = Vocab(emb_filename)
#     # print(len(vocab.embeddings))
#     # exm_idx = vocab.lookup('the', 'en')
#     # print(exm_idx, vocab.embeddings[exm_idx])
#     # exm_idx = vocab.lookup('la', 'de')
#     # print(exm_idx, vocab.embeddings[exm_idx])
#     # exm_idx = vocab.lookup('jpg', 'fr')
#     # print(exm_idx, vocab.embeddings[exm_idx])
#     # exm_idx = vocab.unk_idx
#     # print(exm_idx, vocab.embeddings[exm_idx])



# if __name__ == '__main__':
#     main()
