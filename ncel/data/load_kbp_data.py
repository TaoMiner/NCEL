import os
from ncel.utils.document import *
from ncel.utils.xmlProcessor import *

class KbpDataLoader(xmlHandler):
    def __init__(self, rawtext_path, mention_fname, include_unresolved=False, lowercase=False):
        super(KbpDataLoader, self).__init__()
        self._fpath = rawtext_path
        self._m_fname = mention_fname
        self._include_unresolved = include_unresolved
        self.lowercase = lowercase

    def documents(self):
        all_mentions = dict()
        for (doc_name, mentions) in self.process(self._m_fname):
            postfix_inf = doc_name.rfind(r'.')
            doc_name = doc_name if postfix_inf == -1 else doc_name[:postfix_inf]
            all_mentions[doc_name] = mentions.copy()
        i=0
        for (doc_name, doc_lines) in _WnedFileToDocIterator(self._fpath):