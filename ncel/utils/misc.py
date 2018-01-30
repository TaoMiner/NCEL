# -*- coding: utf-8 -*-
import numpy as np
from collections import deque
import json
from functools import reduce

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
        self.report.append(samples)

    def write_report(self, filename):
        '''Commits the report to a file.'''
        with open(filename, 'w') as f:
            for example in self.report:
                json.dump(example, f, sort_keys=True)
                f.write('\n')

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

# batch_size * node_num , output, np.narray
# batch_size * node_num, y
def ComputeMentionAccuracy(output, y, docs, NIL_thred=0.1):
    batch_docs, max_candidates = y.shape
    doc_acc = 0.0
    total_mentions = 0
    total_mention_correct = 0
    total_candidates = 0
    total_candidates_correct = 0

    for i, doc in enumerate(docs):
        dm_correct = 0
        total_mentions += len(doc.mentions)
        out_doc = output[i,:, :]
        y_doc = y[i,:]
        mc_start = 0
        for j, mention in enumerate(doc.mentions):
            total_candidates += len(mention.candidates)
            mc_end = mc_start + len(mention.candidates)
            output_slice = out_doc[mc_start:mc_end, 0]
            pred = np.argmax(output_slice)
            pred_prob = output_slice[pred]
            y_slice = y_doc[mc_start:mc_end]
            total_candidates_correct += len(output_slice[np.argmax(out_doc[mc_start:mc_end,:], axis=1)==y_slice])
            if (1 not in y_slice and pred_prob <= NIL_thred) or\
                    (1 in y_slice and np.argmax(y_slice)==pred): dm_correct += 1
            mc_start = mc_end
        total_mention_correct += dm_correct
        doc_acc += dm_correct/float(len(doc.mentions))
    return total_candidates, total_candidates_correct, total_mentions, total_mention_correct, batch_docs, doc_acc/float(batch_docs)

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