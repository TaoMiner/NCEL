# -*- coding: utf-8 -*-

class Document:
    def __init__(self, doc_name, doc_id):
        self.name = doc_name
        self.id = doc_id
        self.n_candidates = 0
        self.mentions = []
        self.tokens = []
        self.sentences = []

class Mention:
    def __init__(self, document, mention_start, mention_end, gold_ent_id=None,
                 gold_ent_str=None, is_NIL = False):
        self._document = document
        self._mention_start = mention_start
        self._mention_end = mention_end
        self._gold_ent_id = gold_ent_id
        self._gold_ent_str = gold_ent_str

        self._mention_length = None
        self._mention_str = None

        self._sent_idx = None
        self._pos_in_sent = None

        self._gold_foreign_str = None
        self._gold_foreign_id = None

        self.candidates = None
        self.predicted_sense = None
        self.predicted_sense_global = None

        self._is_trainable = True
        self._is_NIL = is_NIL

    def setStrAndLength(self):
        self._mention_str = self.mention_text()
        self._mention_length = self._mention_end - self._mention_start

    def updateSentIdxByTokenIdx(self):
        token_count = 0
        for i, sent in enumerate(self._document.sentences):
            if token_count + len(sent) >= self._mention_end :
                self._sent_idx = i
                self._pos_in_sent = self._mention_start - token_count
                break
            token_count += len(sent)

    def updateTokenIdxBySentIdx(self):
        offset = sum([len(sent) for sent in self._document.sentences[:self._sent_idx]])
        self._mention_start = offset + self._pos_in_sent
        self._mention_end = self._mention_start + self._mention_length

    def getSent(self):
        if isinstance(self._sent_idx, type(None)) : self.updateSentIdxByTokenIdx()
        return self._document.sentences[self._sent_idx]

    def document(self):
        return self._document

    def gold_ent_str(self):
        return self._gold_ent_str

    def gold_ent_id(self):
        return self._gold_ent_id

    def mention_text(self):
        return ' '.join(self.mention_text_tokenized())

    def mention_text_tokenized(self):
        return self.document().tokens[self._mention_start: self._mention_end]

    def left_context(self, max_len=None, split_by_sent=True):
        l = [t for t in self.left_context_iter(split_by_sent=split_by_sent)]
        l = l if max_len is None or len(l) <= max_len else l[0:max_len]
        l.reverse()
        return l

    def left_context_iter(self, split_by_sent=True):
        if split_by_sent:
            context = self.document().sentences[self._sent_idx]
            start = self._pos_in_sent
        else:
            context = self.document().tokens
            start = self._mention_start
        if start > 0:
            for t in context[start - 1:: -1]:
                yield t

    def right_context(self, max_len=None, split_by_sent=True):
        l = [t for t in self.right_context_iter(split_by_sent=split_by_sent)]
        return l if max_len is None or len(l) <= max_len else l[0:max_len]

    def right_context_iter(self, split_by_sent=True):
        if split_by_sent:
            context = self.document().sentences[self._sent_idx]
            end = self._pos_in_sent + self._mention_length
        else:
            context = self.document().tokens
            end = self._mention_end
        if end < len(context):
            for t in context[end:]:
                yield t

    def left_sent(self, max_len=None):
        l = [t for t in self.left_sent_iter()]
        l = l if max_len is None or len(l) <= max_len else l[0:max_len]
        l.reverse()
        return l

    # t is a list of tokens
    def left_sent_iter(self):
        if self._sent_idx > 0:
            for t in self.document().sentences[self._sent_idx:: -1]:
                yield t

    def right_sent(self, max_len=None):
        l = [t for t in self.right_sent_iter()]
        return l if max_len is None or len(l) <= max_len else l[0:max_len]

    def right_sent_iter(self):
        if self._sent_idx < len(self.document().sentences):
            for t in self.document().sentences[self._sent_idx:]:
                yield t

class Token:
    def __init__(self, text):
        self.text = text
        self.pos = None