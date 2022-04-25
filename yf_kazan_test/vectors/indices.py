import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from collections import Counter

from yf_kazan_test.data import TheDatasets, Datapack

class Vocab:
    def __init__(self, tokens, min_freq):
        self._build_vocab(tokens, min_freq)

    def word2index(self, word):
        return self.word2index[word]

    def index2word(self, index):
        return self.index2word[index]

    def texts_to_indices(self, texts):
        return [[self.word2index[word] for word in line.split() if word in self.word2index] for line in texts]

    def _build_vocab(self, tokens, min_freq):
        self.word2index = {}
        self.index2word = {}
        counts = Counter(tokens)

        idx = 1
        for t in tokens:
            if counts[t] >= min_freq and t not in self.word2index:
                self.word2index[t] = idx
                self.index2word[idx] = t
                idx += 1


def to_indices(source, sentence_len=None, min_freq=1):
    this = to_indices

    def fit_category_indexer():
        this.category_indexer = LabelEncoder()
        this.category_indexer.fit(TheDatasets.train["category_id"])

    def build_vocab():
        tokens = [t for line in TheDatasets.train["about"] for t in line.split()]
        this.vocab = Vocab(tokens, min_freq=min_freq)

    def texts_to_padded_tensor(texts):
        indices = this.vocab.texts_to_indices(texts)
        tensors = [torch.tensor(line).unsqueeze(1) for line in indices]
        padded = pad_sequence(tensors, batch_first=True, padding_value=0).squeeze(2)

        if not sentence_len or sentence_len == padded.shape[1]: 
            return padded

        if sentence_len > padded.shape[1]:
            return torch.cat((padded, torch.zeros((padded.shape[0], sentence_len - padded.shape[1]))), dim=1)
        else:
            return padded[:,:sentence_len]

    fit_category_indexer()
    build_vocab()

    return Datapack(
        X_train = texts_to_padded_tensor(source.X_train["about"]),
        y_train = torch.tensor(this.category_indexer.transform(source.y_train)),
        X_test = texts_to_padded_tensor(source.X_test["about"]),
        y_test = torch.tensor(this.category_indexer.transform(source.y_test))
    )

def decode_category(index_cat):
    return to_indices.category_indexer.inverse_transform(index_cat)


def indices_vocab_size():
    return len(to_indices.vocab.word2index)