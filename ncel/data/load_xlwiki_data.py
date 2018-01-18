import os
from ncel.utils.document import *
import re

def _xlwikiFileToDocIterator(fpath):
    files = os.listdir(fpath)
    for fname in files:
        postfix_inf = fname.rfind(r'.')
        if postfix_inf == -1 : continue
        file_name = fname[:postfix_inf]
        file_type = fname[postfix_inf+1:]
        if file_type == "mentions" or not os.path.exists(os.path.join(fpath, file_name
                    + '.mentions')) or not os.path.exists(os.path.join(fpath, file_name
                    + '.txt')) : continue
        f_txt = open(os.path.join(fpath, file_name+'.txt'), 'r', encoding = 'utf-8')
        txt_lines = f_txt.readlines()
        f_mention = open(os.path.join(fpath, file_name + '.mentions'), 'r', encoding = 'utf-8')
        mention_lines = f_mention.readlines()
        yield (file_name, txt_lines, mention_lines)

class XlwikiDataLoader():
    # degree indicates how difficult the mention is, -1 : all, 0 : easy, 1 : hard
    def __init__(self, path, degree=-1, include_unresolved=False, lowercase=False):
        self._fpath = path
        self._include_unresolved = include_unresolved
        self.lowercase = lowercase
        self.is_hard = degree

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
        i = 0
        for (doc_name, doc_lines, mention_lines) in _xlwikiFileToDocIterator(self._fpath):
            # create doc mention offset index list
            start_inx = dict()
            end_inx = dict()
            tmp_mentions = dict()
            split_inx = set()
            for j, m_line in enumerate(mention_lines):
                items = m_line.strip().split('\t')
                if len(items) < 5 : continue
                doc_start_inx = int(items[0])
                doc_end_inx = int(items[1])
                is_hard = int(items[4])
                if self.is_hard != -1 and is_hard != self.is_hard : continue
                split_inx.add(doc_start_inx)
                split_inx.add(doc_end_inx)
                start_inx[doc_start_inx] = start_inx.get(doc_start_inx, [])
                start_inx[doc_start_inx].append(j)
                end_inx[doc_end_inx] = end_inx.get(doc_end_inx, [])
                end_inx[doc_end_inx].append(j)
                # [en_wiki_name, foreign_wiki_name, new_start_offset, new_tokens_num]
                tmp_mentions[j] = [items[2], items[3], -1, -1]
            # skip those don't have any mention
            if len(tmp_mentions) < 1 : continue
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
                    if line_offset + base_offset in start_inx:
                        for j in start_inx[line_offset + base_offset]:
                            tmp_mentions[j][2] = len(doc.tokens)
                    self._processLineSlice(line_slice, doc, sent)
                    # update mention end index
                    if p + base_offset in end_inx:
                        for j in end_inx[p + base_offset]:
                            tmp_mentions[j][3] = len(doc.tokens)
                            if tmp_mentions[j][2] != -1:
                                m = Mention(doc, tmp_mentions[j][2], tmp_mentions[j][3], gold_ent_str=tmp_mentions[j][0])
                                m._gold_foreign_str = tmp_mentions[j][1]
                                doc.mentions.append(m)

                    if p >= line_len: break
                    line_offset = p
                    split_inx_pos += 1
                if split_inx_pos == len(split_inx) and line_offset < line_len:
                    self._processLineSlice(line[line_offset:], doc, sent)
                base_offset += line_len
                if len(sent) > 0:
                    doc.sentences.append(' '.join(sent))
                del sent[:]
            yield (doc_name, doc)

def load_data(path, degree=-1, lowercase=False):
    print("Loading", path)
    docs = []
    doc_iter = XlwikiDataLoader(path, degree=degree, lowercase=lowercase)
    for doc in doc_iter.documents():
        docs.append(doc)
    return docs

if __name__ == "__main__":
    # Demo:
    docs = XlwikiDataLoader('/Users/ethan/Downloads/xlwikifier-wikidata/data/it/train/')
    for doc_name, doc in docs.documents():
        print(doc.sentences)
        for mention in doc.mentions:
            print(mention.mention_text())
            print(mention.right_context(max_len=5))
        break