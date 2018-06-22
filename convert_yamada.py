import codecs
import re
from ntee.model_reader import ModelReader
import struct
import numpy as np


def loadVocab(file):
    vocab = set()
    with codecs.open(file, 'r', encoding='UTF-8') as fin:
        for line in fin:
            items = re.split(r'\t', line.strip())
            if (items[0] == '</s>'): continue
            vocab.add(items[0])
    print('load vocab of {0} words!'.format(len(vocab)))
    return vocab

# wiki_id \t wiki_label
def loadWikiVocab(filename, id_vocab=None):
    label2id_map = {}
    id2label_map = {}
    with codecs.open(filename, 'r', encoding='UTF-8') as fin:
        for line in fin:
            items = re.split(r'\t', line.strip())
            if len(items) < 2 or len(items[0].strip()) < 1 or len(items[1].strip()) < 1 or\
                    (not isinstance(id_vocab, type(None)) and items[0] not in id_vocab): continue
            label2id_map[items[1]] = items[0]
            id2label_map[items[0]] = items[1]
    return label2id_map, id2label_map

def saveVector(filename, vocab_size, layer_size, vectors):
    with codecs.open(filename, 'w') as fout:
        fout.write("{0} {1}\n".format(vocab_size, layer_size))
        for label in vectors:
            fout.write("{0}\t".format(label.encode('utf-8', 'ignore')))
            for i in range(layer_size):
                fout.write(struct.pack('f', vectors[label][i]))
            fout.write('\n')

word_vocab_file = '/data/caoyx/etc/vocab/envocab/vocab_word.txt'
entity_vocab_file = '/data/caoyx/etc/vocab/envocab/vocab_entity.txt'
wiki_vocab_file = '/data/caoyx/el_datasets/vocab_entity.dat'
word_vec_file = '/data/caoyx/etc/yamada_vec/vectors_word0'
entity_vec_file = '/data/caoyx/etc/yamada_vec/vectors_entity0'
weight_file = '/data/caoyx/etc/yamada_vec/vectors_weight.npy'
bias_file = '/data/caoyx/etc/yamada_vec/vectors_bias.npy'

model = ModelReader('/data/caoyx/etc/yamada_vec/ntee_300_sentence.joblib')

word_vocab = loadVocab(word_vocab_file)
entity_vocab = loadVocab(entity_vocab_file)
_, id2label_vocab = loadWikiVocab(wiki_vocab_file, id_vocab=entity_vocab)
w2v = {}
loaded = 0
for w in word_vocab:
    w_vec = model.get_word_vector(w)
    if w_vec is not None:
        w2v[w] = w_vec
        loaded += 1
print("load {} words! saving ...".format(loaded))
saveVector(word_vec_file, loaded, 300, w2v)

e2v = {}
loaded = 0
for e in entity_vocab:
    if e not in id2label_vocab: continue
    el = id2label_vocab[e]
    e_vec = model.get_entity_vector(el)
    if e_vec is not None:
        e2v[e] = e_vec
        loaded += 1
print("load {} entities! saving ...".format(loaded))
saveVector(entity_vec_file, loaded, 300, e2v)

w = model._W
b = model._b
np.save(weight_file, w)
np.save(bias_file, b)