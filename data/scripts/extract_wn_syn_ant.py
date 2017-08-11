from nltk.corpus import wordnet as wn
import os
import numpy as np
import sys
import pickle

def get_syn(words):
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
    print("extracted synonyms, mean %s, median %s, max %s" % (np.mean(vals), np.median(vals), max(vals)))
    return syns


def get_ant(words):
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
                    ant_syn = list(get_syn([ll]).values())[0]
                    for a_s in ant_syn:
                        if a_s in words:
                            sl.append(a_s)
            synlist.append(sl)
        ants[w] = list(set(sum(synlist, [])))
    vals = list(map(lambda x: len(x), ants.values()))
    print("extracted antonyms, mean %s, median %s, max %s" % (np.mean(vals), np.median(vals), max(vals)))
    return ants


def read_vocab(path):
    words = []
    with open(path, 'r') as f:
        for l in f.readlines():
            words.append(l.split(' ')[0][2:-1]) # drop byte symbol and quotes

    return words[1:] # skip UNK

if __name__ == '__main__':
    path = sys.argv[-1]
    words = read_vocab(os.path.join(path, 'vocab.txt'))
    with open(os.path.join(path, 'vocab_ant.pickle'), 'wb') as f:
        pickle.dump(get_ant(words), f)
    with open(os.path.join(path, 'vocab_syn.pickle'), 'wb') as f:
        pickle.dump(get_syn(words), f)
