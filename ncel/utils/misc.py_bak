# -*- coding: utf-8 -*-
import numpy as np
from collections import deque
import json
from functools import reduce
import re

def flatten(l):
    if hasattr(l, '__len__'):
        return reduce(lambda x, y: x + flatten(y), l, [])
    else:
        return [l]

def time_per_token(num_tokens, total_time):
    return sum(total_time) / float(sum(num_tokens))

class EvalReporter(object):
    def __init__(self):
        super(EvalReporter, self).__init__()
        self.report = []

    def save_batch(self, samples):
        self.report.extend(samples)

    def write_report(self, filename):
        '''Commits the report to a file.'''
        with open(filename, 'w') as f:
            for line in self.report:
                f.write('{}\n'.format(line))

class Accumulator(object):
    """Accumulator. Makes it easy to keep a trailing list of statistics."""

    def __init__(self, maxlen=None):
        self.maxlen = maxlen
        self.cache = dict()

    def add(self, key, val):
        self.cache.setdefault(key, deque(maxlen=self.maxlen)).append(val)

    def get(self, key, clear=True):
        ret = self.cache.get(key, [])
        if clear:
            try:
                del self.cache[key]
            except BaseException:
                pass
        return ret

    def get_avg(self, key, clear=True):
        return np.array(self.get(key, clear)).mean()

def recursively_set_device(inp, gpu):
    if hasattr(inp, 'keys'):
        for k in list(inp.keys()):
            inp[k] = recursively_set_device(inp[k], gpu)
    elif isinstance(inp, list):
        return [recursively_set_device(ii, gpu) for ii in inp]
    elif isinstance(inp, tuple):
        return (recursively_set_device(ii, gpu) for ii in inp)
    elif hasattr(inp, 'cpu'):
        if gpu >= 0:
            inp = inp.cuda()
        else:
            inp = inp.cpu()
    return inp

# output: batch_size * cand_num
# y: batch_size
def ComputeAccuracy(output, y, docs, include_unresolved=False):
    total_mentions, max_cand_num = output.size()
    batch_docs = len(docs)

    # get the index of the max log-probability, batch_size
    pred = output.max(1, keepdim=False)[1].cpu()
    # batch_size
    mention_correct = pred.eq(y)
    doc_acc = 0.0
    s = 0
    for i, doc in enumerate(docs):
        e = s + len(doc.mentions)
        d_mention_slice = mention_correct[s:e]
        doc_acc += d_mention_slice.sum() / float(len(d_mention_slice))
        s = e
    total_mention_correct = mention_correct.sum()

    return total_mentions, total_mention_correct, batch_docs, doc_acc/float(batch_docs)

# wiki_id \t wiki_label
def loadWikiVocab(filename, id_vocab=None):
    label2id_map = {}
    id2label_map = {}
    with open(filename, 'r', encoding='UTF-8') as fin:
        for line in fin:
            items = re.split(r'\t', line.strip())
            if len(items) < 2 or len(items[0].strip()) < 1 or len(items[1].strip()) < 1 or\
                    (not isinstance(id_vocab, type(None)) and items[0] not in id_vocab): continue
            label2id_map[items[1]] = items[0]
            id2label_map[items[0]] = items[1]
    return label2id_map, id2label_map

# redirect_id \t ent_id
def loadRedirectVocab(filename, id_vocab=None):
    redirectid_map = {}
    with open(filename, 'r', encoding='UTF-8') as fin:
        for line in fin:
            items = re.split(r'\t', line.strip())
            if len(items) < 2 or len(items[0].strip()) < 1 or len(items[1].strip()) < 1 or\
                    (not isinstance(id_vocab, type(None)) and items[0] not in id_vocab): continue
            redirectid_map[items[0]] = items[1]
    return redirectid_map

def loadStopWords(filename):
    stop_words = set()
    with open(filename, 'r', encoding='UTF-8') as fin:
        for line in fin:
            stop_words.add(line.strip())
    return stop_words


def inspectDoc(doc, word_vocab=None):
    if not isinstance(word_vocab, type(None)):
        word_label_vocab = dict(
            [(word_vocab[key], key) for key in word_vocab if key in word_vocab])
        tokens = [word_label_vocab[t] for t in doc.tokens]
        sentences = [[word_label_vocab[t] for t in s] for s in doc.sentences]
    else:
        tokens = doc.tokens
        sentences = doc.sentences
    print(doc.name)
    print(tokens)
    print(sentences)
    for m in doc.mentions:
        if m._sent_idx is None : m.updateSentIdxByTokenIdx()
        sent_ids = doc.sentences[m._sent_idx][m._pos_in_sent:m._pos_in_sent+m._mention_length]
        if not isinstance(word_vocab, type(None)):
            print("ment_str:{}, m_len:{}, m_pos:{}, m_sent_pos:{}.".format(m._mention_str, m._mention_length,
                 ' '.join(tokens[m._mention_start:m._mention_end]),
                 ' '.join([word_label_vocab[s] for s in sent_ids])))
            print("left_contexts:{}.".format(' '.join([word_label_vocab[w] for w in m.left_context(max_len=5)])))
            print("right_contexts:{}.".format(' '.join([word_label_vocab[w] for w in m.right_context(max_len=5)])))
            print("left_sent:{}.".format(' '.join([word_label_vocab[w] for s in m.left_sent(max_len=1) for w in s])))
            print("right_sent:{}.".format(' '.join([word_label_vocab[w] for s in m.right_sent(max_len=1) for w in s])))
        else:
            print("ment_str:{}, m_len:{}, m_pos:{}, m_sent_pos:{}.".format(m._mention_str, m._mention_length,
                ' '.join(tokens[m._mention_start:m._mention_end]), ' '.join(sent_ids)))
            print("left_contexts:{}.".format(' '.join(m.left_context(max_len=5))))
            print("right_contexts:{}.".format(' '.join(m.right_context(max_len=5))))
            print("left_sent:{}.".format(' '.join([w for s in m.left_sent(max_len=1) for w in s])))
            print("right_sent:{}.".format(' '.join([w for s in m.right_sent(max_len=1) for w in s])))


# contexts1 : batch * cand_num * tokens
# contexts2 : batch * cand_num * tokens
# base_feature : batch * cand_num * features
# candidates : batch * cand_num
# candidates_entity: batch * cand_num
# length: batch
# truth: batch
def inspectBatch(batch, vocab, docs, only_one=True):
    base, context1, context2, m_strs, cids, cids_entity, num_candidates, num_mentions, y = batch
    word_vocab, entity_vocab, sense_vocab, id2wiki_vocab = vocab
    # reverse vocab
    word_label_vocab = dict(
        [(word_vocab[key], key) for key in word_vocab if key in word_vocab])
    sense_id_vocab = dict(
        [(sense_vocab[key], key) for key in sense_vocab if key in sense_vocab])
    entity_id_vocab = dict(
        [(entity_vocab[key], key) for key in entity_vocab if key in entity_vocab])

    batch_size, cand_num = cids.shape
    print("batch_size:{}, candidate num:{}".format(batch_size, cand_num))
    show_cand_num = -1
    for i in range(batch_size):
        inspectDoc(docs[i], word_vocab=word_vocab)
        count = 0
        print("num_cand:{}, truth:{}".format(num_candidates[i], y[i]))
        for j in range(cand_num):
            print("base:{}".format(base[i][j]))
            print("context1:{}".format(' '.join([word_label_vocab[w] for w in context1[i][j]])))
            if context2 is not None:
                print("context2:{}".format(' '.join([word_label_vocab[w] for w in context2[i][j]])))
            sense_label = id2wiki_vocab[sense_id_vocab[cids[i][j]]] if sense_id_vocab[cids[i][j]] in id2wiki_vocab else ''
            print("cids:{}".format(sense_label))
            entity_label = id2wiki_vocab[entity_id_vocab[cids_entity[i][j]]] if entity_id_vocab[
                                           cids_entity[i][j]] in id2wiki_vocab else ''
            print("cids_entity:{}".format(entity_label))
            count+=1
            if show_cand_num>0 and count == show_cand_num: break

        if only_one: break