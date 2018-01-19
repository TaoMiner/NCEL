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

en_punctuation = " \'\",:()\-\n"
zh_punctuation = " ·＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"
en_sent_split_punc = "\.?!"
zh_ssplit_punc = "！？｡。"
punction = '[{0}{1}]'.format(en_punctuation, zh_punctuation)
ssplict_puncRE = re.compile('[{0}{1}]'.format(en_sent_split_punc, zh_ssplit_punc))

class XlwikiDataLoader():
    # genre indicates how difficult the mention is, 0 : easy, 1 : hard, 2 : all
    def __init__(self, path, genre=2, lowercase=False):
        self._fpath = path
        self.lowercase = lowercase
        self.is_hard = genre

    def _processLineSlice(self, line_slice, doc, sent):
        # split words in line slice
        # preprocess
        raw_tokens = re.split(punction, line_slice)

        for rt in raw_tokens:
            if len(rt) < 1 : continue
            m = ssplict_puncRE.search(rt)
            dot_idx = m.start() if m else -1
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
                if self.is_hard in [0, 1] and is_hard != self.is_hard : continue
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

def load_data(text_path=None, mention_file=None, kbp_id2wikiid_file=None, genre=0, include_unresolved=False, lowercase=False):
    assert not isinstance(type(text_path), None), "xlwiki data requires raw path!"
    print("Loading", text_path)
    docs = []
    doc_iter = XlwikiDataLoader(text_path, genre=genre, lowercase=lowercase)
    for doc in doc_iter.documents():
        docs.append(doc)
    return docs

if __name__ == "__main__":
    # Demo:
    docs = load_data(text_path='/Users/ethan/Downloads/xlwikifier-wikidata/data/it/train/')
    print(docs[0].doc_name)
    print(docs[0].mentions)
    print(docs[0].tokens)