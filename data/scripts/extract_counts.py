import os
import numpy as np
import pandas as pd
import sys
import pickle
import spacy
from spacy.attrs import ORTH
from itertools import product
from collections import Counter


def extract_counts(vocab, text, contexts):
    idword = {word: j for j, word in enumerate(vocab)}
    print("counting single")
    D = Counter(text)
    single_counts = {idword[w]: i for w, i  in D.items() if w in idword}
    single_counts = np.array(list(map(lambda y: y[1], sorted(single_counts.items(), key=lambda x: x[0]))))

    keys = list(idword.values())
    ars = []

    print("counting double")
    for i, (w, c) in enumerate(contexts.items()):
        C = Counter(c)
        t = []
        for k in keys:
            t.append(C[k])
        ars.append(t)

    double_counts = np.array(ars)
    return single_counts, double_counts



def read_vocab(vocab_fd):
    return list(map(lambda x: x.split(' ')[0], vocab_fd.readlines()))


def build_vocab(text_fd, vocab_fd):
    doc = text_fd.read().split()
    counts = Counter(doc)
    print(len(counts), 'unique words in corpus')
    for word_id in doc:
        if counts[word_id] >= 100:
            vocab_fd.write(word_id + ' ' + str(counts[word_id]) + '\n')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: extract_counts.py corpus_path.txt')
        exit()
    path = sys.argv[-1]
    path_dir = os.path.dirname(path)

    vocab_name = 'vocab.txt'
    if not os.path.isfile(os.path.join(path_dir, vocab_name)):
        with open(os.path.join(path_dir, vocab_name), 'w') as vocab_fd, open(path, 'r') as text_fd:
            build_vocab(text_fd, vocab_fd)

    with open(os.path.join(path_dir, vocab_name), 'r') as vocab_fd:
        words = read_vocab(vocab_fd)

    with open(os.path.join(path_dir, 'context.pickle'), 'rb') as f, open(path, 'r') as text_fd:
        contexts = pickle.load(f)
        text = text_fd.read().split()
        sc, dc = extract_counts(words, text, contexts)
        print(sc.shape, dc.shape)
        pd.Series(sc).to_pickle(os.path.join(path_dir, 'single_counts.pickle'))
        pd.DataFrame(dc).to_pickle(os.path.join(path_dir, 'double_counts.pickle'))
