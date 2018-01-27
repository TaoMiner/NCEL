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

def ComputeMentionAccuracy(output, y, max_candidates, docs, NIL_thred=0.1):
    batch_docs = len(docs)
    doc_acc = 0.0
    total_mentions = 0
    total_mention_correct = 0
    mc_start = 0
    for i, doc in enumerate(docs):
        dm_correct = 0
        total_mentions += len(doc.mentions)
        for j, mention in enumerate(doc.mentions):
            mc_end = mc_start + len(mention.candidates)
            output_slice = output[mc_start:mc_end, 0]
            pred = np.argmax(output_slice)
            pred_prob = output_slice[pred]
            y_slice = y[mc_start:mc_end, 0]
            if (1 not in y_slice and pred_prob <= NIL_thred) or\
                    (1 in y_slice and np.argmax(y_slice)==pred): dm_correct += 1
            mc_start = mc_end
        mc_start = i*max_candidates
        total_mention_correct += dm_correct
        doc_acc += dm_correct/float(len(doc.mentions))
    return total_mentions, total_mention_correct, batch_docs, doc_acc/float(batch_docs)
