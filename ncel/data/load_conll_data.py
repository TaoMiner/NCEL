# -*- coding: utf-8 -*-
from ncel.utils.document import *
import re

DOC_GENRE = ('testa', 'testb', 'train')

def _CoNLLFileToDocIterator(fname, split='testa'):
    f = open(fname,'r')
    lines = f.readlines()

    curdocName = None
    curdocSplit = None
    curdoc = None

    for line in lines:
        line = line.strip()
        if line.startswith('-DOCSTART-'):
            if curdocName is not None and curdocSplit == split:
                yield (curdoc, curdocName)
            sp = line.split(' ')
            curdocName = sp[2][:-1]
            curdocSplit = DOC_GENRE[0] if sp[1].endswith(DOC_GENRE[0]) else (DOC_GENRE[1] if sp[1].endswith(DOC_GENRE[1]) else DOC_GENRE[2])
            curdoc = []
        else:
            curdoc.append(line)
    if curdocName is not None and curdocSplit == split:
        yield (curdoc, curdocName)

class CoNLLIterator:
    # genre \in [0, 1, 2] indicates ['testa', 'testb', 'train']
    def __init__(self, fname, genre=0, include_unresolved=False, lowercase=False):
        self._fname = fname
        if genre not in [0, 1, 2] : genre = 0
        self._split = DOC_GENRE[genre]
        self._include_unresolved = include_unresolved
        self.lowercase = lowercase

    def documents(self):
        i = 0
        for (doc_lines, doc_name) in _CoNLLFileToDocIterator(self._fname, self._split):
            doc = Document(doc_name, i)
            i += 1
            mention = None
            sent = []
            for line in doc_lines:
                if len(line) == 0:
                    # sentence boundary.
                    doc.sentences.append(sent)
                    sent = []
                    continue
                t = line.split('\t')
                if self.lowercase: t[0] = t[0].lower()
                sent.append(t[0])
                doc.tokens.append(t[0])
                if len(t) == 1:
                    continue

                if t[1] != 'I' and mention is not None:
                    mention = None
                if t[1] == 'I' and (t[3] != '--NME--' or self._include_unresolved):
                    mention._mention_end += 1

                if t[1] == 'B' and (t[3] != '--NME--' or self._include_unresolved):
                    if t[3] != '--NME--':
                        gold_ent_str = re.sub(r'_', ' ', t[3])
                        gold_ent_id = t[5]
                        mention = Mention(doc, len(doc.tokens) - 1, len(doc.tokens),
                                          gold_ent_id=gold_ent_id, gold_ent_str=gold_ent_str)
                    else:
                        mention = Mention(doc, len(doc.tokens) - 1, len(doc.tokens), is_NIL=True)
                    doc.mentions.append(mention)
            for i, m in enumerate(doc.mentions):
                doc.mentions[i].setStrAndLength()
            yield doc

    def mentions(self):
        for doc in self.documents():
            for mention in doc.mentions:
                yield mention

def load_data(text_path=None, mention_file=None, kbp_id2wikiid_file=None, genre=0,
              include_unresolved=False, lowercase=False, wiki_entity_file=None):
    assert not isinstance(mention_file, type(None)), "conll data requires mention file!"
    print("Loading", mention_file)
    docs = []
    doc_iter = CoNLLIterator(mention_file, genre=genre, include_unresolved = include_unresolved, lowercase=lowercase)
    for doc in doc_iter.documents():
        docs.append(doc)
    return docs

if __name__ == "__main__":
    # Demo:
    docs = load_data(mention_file='/home/caoyx/data/conll/AIDA-YAGO2-dataset.tsv')
    print(docs[0].name)
    for m in docs[0].mentions:
        print("{0}, {1}, {2}".format(m._mention_start, m._mention_end, m._gold_ent_id))
    print(docs[0].tokens)