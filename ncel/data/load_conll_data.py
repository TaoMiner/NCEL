from ncel.utils.document import *

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
            if curdocName is not None and \
                    (split == 'all' or curdocSplit == split):
                yield (curdoc, curdocName, curdocSplit)
            sp = line.split(' ')
            curdocName = sp[2][:-1]
            curdocSplit = DOC_GENRE[0] if sp[1].endswith(DOC_GENRE[0]) else (DOC_GENRE[1] if sp[1].endswith(DOC_GENRE[1]) else DOC_GENRE[2])
            curdoc = []
        else:
            curdoc.append(line)
    if curdocName is not None and (split == 'all' or curdocSplit == split):
        yield (curdoc, curdocName)


def _CoNLLRawToTuplesIterator(lines):
    '''
    yields tuples:
    (surface,ismention,islinked,YAGO2,WikiURL,WikiId,FB)
    surface is either a word or the full mention
    '''
    for line in lines:
        if len(line) == 0:
            # sentence boundary.
            continue
        t = line.split('\t')
        if len(t) == 1:
            yield (t[0], t[0], False, None, None, None, None, None)
        else:
            if t[1] != 'B':
                continue
            if t[3] == '--NME--':
                yield (t[2], True, False, None, None, None, None)
            else:
                yield (t[2], True, True, t[3], t[4], int(t[5]), t[6] if len(t) >= 7 else None)

class CoNLLIterator:
    # the new iterator does not support using a zip file.
    def __init__(self, fname, genre='testa', include_unresolved=False, lowercase=False):
        self._fname = fname
        self._split = genre
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
                if self.lowercase:
                    line = line.lower()
                if len(line) == 0:
                    # sentence boundary.
                    doc.sentences.append(' '.join(sent))
                    sent = []
                    continue
                t = line.split('\t')

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
                        gold_ent_str = t[3]
                        gold_ent_id = t[5]
                        mention = Mention(doc, len(doc.tokens) - 1, len(doc.tokens),
                                          gold_ent_id=gold_ent_id, gold_ent_str=gold_ent_str)
                    else:
                        mention = Mention(doc, len(doc.tokens) - 1, len(doc.tokens))
                    doc.mentions.append(mention)

            yield doc

    def mentions(self):
        for (doc, doc_genre) in self.documents():
            for mention in doc.mentions:
                yield mention

def load_data(text_path, mention_path=None, genre='testa', lowercase=False):
    print("Loading", text_path)
    docs = []
    doc_iter = CoNLLIterator(text_path, genre=genre, lowercase=lowercase)
    for doc in doc_iter.documents():
        docs.append(doc)
    return docs

if __name__ == "__main__":
    # Demo:
    docs = load_data('/data/CoNLL/CoNLL_AIDA-YAGO2-dataset.tsv')
    print(docs[0])