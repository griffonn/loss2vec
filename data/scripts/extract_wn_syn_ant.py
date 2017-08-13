
from nltk.corpus import wordnet as wn
import os
import numpy as np
import sys
import pickle
import spacy
from spacy.attrs import ORTH
from itertools import product


def get_syn(words, quiet=False):
    syns = {}
    for w in words:
        synlist = []
        for ss in wn.synsets(w):
            sl = []
            for l in ss.lemmas():
                ll = l.name()
                if ll in words:
                    sl.append(ll)
            synlist.append(sl)
        syns[w] = list(set(sum(synlist, [])))
    vals = list(map(lambda x: len(x), syns.values()))
    if not quiet:
        print("extracted synonyms, mean %s, median %s, max %s" % (np.mean(vals), np.median(vals), max(vals)))
        for i in range(max(vals)):
            print('At least %d synonyms: %d' % (i, len(list(filter(lambda x: x >= i, vals)))))
    return syns, max(vals)


def get_ant(words, quiet=False):
    ants = {}
    for w in words:
        synlist = []
        for ss in wn.synsets(w):
            sl = []
            for l in ss.lemmas():
                for a in l.antonyms():
                    ll = a.name()
                    if ll in words:
                        sl.append(ll)
                    ant_syn = list(get_syn([ll], True)[0].values())[0]
                    if len(ant_syn) > 1:
                        print(len(ant_syn))
                    for a_s in ant_syn:
                        if a_s in words:
                            sl.append(a_s)
            synlist.append(sl)
        ants[w] = list(set(sum(synlist, [])))
    vals = list(map(lambda x: len(x), ants.values()))
    if not quiet:
        print("extracted antonyms, mean %s, median %s, max %s" % (np.mean(vals), np.median(vals), max(vals)))
        for i in range(max(vals)):
            print('At least %d antonyms: %d' % (i, len(list(filter(lambda x: x >= i, vals)))))
    return ants, max(vals)


def read_vocab(vocab_fd):
    return list(map(lambda x: x.split(' ')[0], vocab_fd.readlines()))


def get_context(text, i, window_size):
    return text[i-1-window_size:i-1].tolist() + text[i+1:i+1+window_size].tolist()


def extract_labels(vocab, text_fd, window_size):
    import time
    start = time.time()
    text = text_fd.read().split()

    print('text length:', len(text))
    print('vocab length:', len(vocab))

    idword = {word: j for j, word in enumerate(vocab)}
    idtext = list(map(lambda x: idword[x] if x in idword else -1, text))

    print("creating array")
    doc = np.array(list(zip([
        idtext[:-10],
        idtext[1:-9],
        idtext[2:-8],
        idtext[3:-7],
        idtext[4:-6],
        idtext[5:-5],
        idtext[6:-4],
        idtext[7:-3],
        idtext[8:-2],
        idtext[9:-1],
        idtext[10:]
    ]))).squeeze()

    labels = {}

    for j, word in enumerate(idword.values()):
        if not j % 1000: print(j)
        a = doc[:, np.where(doc[5, :] == word)[0]][[0,1,2,3,4,6,7,8,9,10], :]
        labels[word] = a.reshape([1, a.shape[0] * a.shape[1]]).tolist()[0]

    print(time.time() - start)

    # vals = list(map(lambda x: len(x), labels.values()))
    # print("extracted contexts, mean %s, median %s, max %s" % (np.mean(vals), np.median(vals), max(vals)))
    return labels


def build_vocab(text_fd, vocab_fd):
    nlp = spacy.load('en_core_web_md')
    doc = nlp(str(text_fd.read()))
    counts = doc.count_by(ORTH)
    print(len(counts), 'unique words in corpus')
    for word_id, count in sorted(counts.items(), reverse=True, key=lambda item: item[1]):
        if count >= 100:
            vocab_fd.write(nlp.vocab.strings[word_id] + ' ' + str(count) + '\n')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: extract_wn_syn_ant.py corpus_path.txt')
        exit()
    path = sys.argv[-1]
    path_dir = os.path.dirname(path)

    vocab_name = 'vocab.txt'
    if not os.path.isfile(os.path.join(path_dir, vocab_name)):
        with open(os.path.join(path_dir, vocab_name), 'w') as vocab_fd, open(path, 'rb') as text_fd:
            build_vocab(text_fd, vocab_fd)

    with open(os.path.join(path_dir, vocab_name), 'r') as vocab_fd:
        words = read_vocab(vocab_fd)

    ant_name = 'ant.pickle'
    if not os.path.isfile(os.path.join(path_dir, ant_name)):
        ants, max_ants = get_ant(words)
        with open(os.path.join(path_dir, ant_name), 'wb') as f:
            pickle.dump(ants, f)
    else:
        with open(os.path.join(path_dir, ant_name), 'rb') as f:
            ants = pickle.load(f)
            max_ants = max(map(lambda x: len(x), ants.values()))

    syn_name = 'syn.pickle'
    if not os.path.isfile(os.path.join(path_dir, syn_name)):
        syns, max_syns = get_syn(words)
        with open(os.path.join(path_dir, 'syn.pickle'), 'wb') as f:
            pickle.dump(syns, f)
    else:
        with open(os.path.join(path_dir, syn_name), 'rb') as f:
            syns = pickle.load(f)
            max_syns = max(map(lambda x: len(x), syns.values()))

    # for pair in product(range(max_ants, max_syns)):
    #     print(pair)
    with open(os.path.join(path_dir, 'context.pickle'), 'wb') as f, open(path, 'r') as text_fd:
        pickle.dump(extract_labels(words, text_fd, 5), f)
