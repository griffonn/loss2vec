from nltk.corpus import wordnet as wn
import os
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
            synlist.append(sl)
        ants[w] = list(set(sum(synlist, [])))
    return ants


def read_vocab_binary(path):
    words = []
    with open(path, 'r') as f:
        for l in f.readlines():
            words.append(l.split(' ')[0][2:-1]) # drop byte symbol and quotes

    return words[1:] # skip UNK

if __name__ == '__main__':
    path = sys.argv[-1]
    words = read_vocab_binary(os.path.join(path, 'vocab.txt'))
    with open(os.path.join(path, 'vocab_ant.pickle'), 'wb') as f:
        pickle.dump(get_ant(words), f)
    with open(os.path.join(path, 'vocab_syn.pickle'), 'wb') as f:
        pickle.dump(get_syn(words), f)