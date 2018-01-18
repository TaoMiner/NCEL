import unicodedata

class Document:
    def __init__(self, doc_name, doc_id):
        self.name = doc_name
        self.id = doc_id
        self.tokens = []
        self.mentions = []
        self.sentences = []

class Mention:
    def __init__(self, document, mention_start, mention_end, gold_ent_id=None, gold_ent_str=None):
        self._document = document
        self._mention_start = mention_start
        self._mention_end = mention_end
        self._gold_ent_id = gold_ent_id
        self._gold_ent_str = gold_ent_str

        self._gold_foreign_str = None
        self._gold_foreign_id = None

        self.candidates = None
        self.predicted_sense = None
        self.predicted_sense_global = None

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

    def left_context(self, max_len=None):
        l = [t for t in self.left_context_iter()]
        l = l if max_len is None or len(l) <= max_len else l[0:max_len]
        l.reverse()
        return l

    def left_context_iter(self):
        if self._mention_start > 0:
            for t in self.document().tokens[self._mention_start - 1:: -1]:
                yield t

    def right_context(self, max_len=None):
        l = [t for t in self.right_context_iter()]
        return l if max_len is None or len(l) <= max_len else l[0:max_len]

    def right_context_iter(self):
        if self._mention_end < len(self.document().tokens):
            for t in self.document().tokens[self._mention_end:]:
                yield t


class Token:
    def __init__(self, text):
        self.text = text
        self.pos = None