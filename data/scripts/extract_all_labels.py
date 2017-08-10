import sys
import pickle


def read_vocab(path):
    words = []
    with open(path, 'r') as f:
        for l in f.readlines():
            words.append(l.split(' ')[0][2:-1]) # drop byte symbol and quotes

    return words[1:] # skip UNK


def get_context(text, i, window_size):
    return text[i-window_size:i-1] + text[i+1:i+1+window_size]


def extract_labels(vocab, corpus_path, window_size):
    with open(corpus_path, 'r') as c:
        text = c.read().split(' ')
        labels = {}
        for i, word in enumerate(text):
            if word in vocab:
                if word not in labels:
                    labels[word] = []
                labels[word] += filter(lambda x: x in vocab, get_context(text, i, window_size))
    return labels

if __name__ == '__main__':
    text_path, vocab_path, pickle_path, window_size = sys.argv[-4:]
    window_size = int(window_size)
    words = read_vocab(vocab_path)
    labels = extract_labels(words, text_path, window_size)
    with open(pickle_path, 'wb') as p:
        pickle.dump(labels, p)