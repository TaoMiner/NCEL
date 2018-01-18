from ncel.utils.document import *
from ncel.utils.xmlProcessor import *
import os
import re

def _WnedFileToDocIterator(fpath):
    doc_name = None

    files = os.listdir(fpath)
    for fname in files:
        postfix_inf = fname.rfind(r'.')
        doc_name = fname if postfix_inf == -1 else fname[:postfix_inf]
        f = open(os.path.join(fpath, fname), 'r')
        lines = f.readlines()
        yield (doc_name, lines)

class WnedDataLoader(xmlHandler):
    def __init__(self, rawtext_path, mention_fname, include_unresolved=False, lowercase=False):
        super(WnedDataLoader, self).__init__()
        self._fpath = rawtext_path
        self._m_fname = mention_fname
        self._include_unresolved = include_unresolved
        self.lowercase = lowercase

    def _processLineSlice(self, line_slice, doc, sent):
        # split words in line slice
        # preprocess
        raw_tokens = re.split(r'[ \'",:()\-\n]', line_slice)

        for rt in raw_tokens:
            if len(rt) < 1 : continue
            dot_idx = rt.find('.')
            if dot_idx < 0 :
                sent.append(rt)
                doc.tokens.append(rt)
            elif dot_idx == 0:
                doc.tokens.append(rt[dot_idx+1:])
                doc.sentences.append(' '.join(sent))
                del sent[:]
                sent.append(rt[dot_idx+1:])
            elif dot_idx == len(rt)-1:
                doc.tokens.append(rt[:dot_idx])
                sent.append(rt[:dot_idx])
                doc.sentences.append(' '.join(sent))
                del sent[:]
            else:
                doc.tokens.append(rt[:dot_idx])
                doc.tokens.append(rt[dot_idx+1:])
                sent.append(rt[:dot_idx])
                doc.sentences.append(' '.join(sent))
                del sent[:]
                sent.append(rt[dot_idx + 1:])

    def documents(self):
        all_mentions = dict()
        for (doc_name, mentions) in self.process(self._m_fname):
            postfix_inf = doc_name.rfind(r'.')
            doc_name = doc_name if postfix_inf == -1 else doc_name[:postfix_inf]
            all_mentions[doc_name] = mentions.copy()
        i=0
        for (doc_name, doc_lines) in _WnedFileToDocIterator(self._fpath):
            if doc_name not in all_mentions : continue
            # create doc mention offset index list
            start_inx = dict()
            end_inx = dict()
            tmp_mentions = dict()
            split_inx = set()
            doc_mentions = all_mentions[doc_name]
            for j, doc_mention in enumerate(doc_mentions):
                doc_start_inx = doc_mention['offset']
                doc_end_inx = doc_mention['offset'] + doc_mention['length']
                split_inx.add(doc_start_inx)
                split_inx.add(doc_end_inx)
                start_inx[doc_start_inx] = start_inx.get(doc_start_inx, [])
                start_inx[doc_start_inx].append(j)
                end_inx[doc_end_inx] = end_inx.get(doc_end_inx, [])
                end_inx[doc_end_inx].append(j)
                # [_, _, new_start_offset, new_tokens_num]
                tmp_mentions[j] = [doc_mention['mention'], doc_mention['wiki_name'], -1, -1]

            # sort the slice inx
            split_inx = sorted(split_inx)
            split_inx_pos = 0
            # processed line length
            base_offset = 0

            doc = Document(doc_name, i)
            i += 1
            sent = []
            for line in doc_lines:
                if self.lowercase: line = line.lower()

                line_offset = 0
                line_len = len(line)

                for p in split_inx[split_inx_pos:]:
                    p -= base_offset
                    line_slice = line[line_offset:p] if p < line_len else line[line_offset:]
                    # process line segment, whose boundries are the annotations
                    # update mention start index
                    if line_offset+base_offset in start_inx:
                        for j in start_inx[line_offset+base_offset]:
                            tmp_mentions[j][2] = len(doc.tokens)
                    self._processLineSlice(line_slice, doc, sent)
                    # update mention end index
                    if p + base_offset in end_inx:
                        for j in end_inx[p + base_offset]:
                            tmp_mentions[j][3] = len(doc.tokens)
                            if tmp_mentions[j][2] != -1:
                                m = Mention(doc, tmp_mentions[j][2], tmp_mentions[j][3])
                                if tmp_mentions[j][1] != "NIL": m._gold_ent_str = tmp_mentions[j][1]
                                doc.mentions.append(m)

                    if p >= line_len : break
                    line_offset = p
                    split_inx_pos += 1
                if split_inx_pos == len(split_inx) and line_offset < line_len:
                    self._processLineSlice(line[line_offset:], doc, sent)
                base_offset += line_len
                if len(sent) > 0:
                    doc.sentences.append(' '.join(sent))
                del sent[:]
            yield (doc_name, doc)

    def mentions(self):
        for (doc_name, doc) in self.documents():
            for mention in doc.mentions:
                yield mention

def load_data(text_path, mention_path=None, lowercase=False):
    print("Loading", text_path)
    docs = []
    doc_iter = WnedDataLoader(text_path, mention_fname=mention_path, lowercase=lowercase)
    for doc in doc_iter.documents():
        docs.append(doc)
    return docs

if __name__ == "__main__":
    # Demo:
    docs = WnedDataLoader('/Users/ethan/Downloads/WNED/wned-datasets/ace2004/RawText/', '/Users/ethan/Downloads/WNED/wned-datasets/ace2004/ace2004.xml')
    for doc_name, doc in docs.documents():
        print(doc.sentences)
        for mention in doc.mentions:
            print(mention.mention_text())
            print(mention.right_context(max_len=5))
        break