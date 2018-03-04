# -*- coding: utf-8 -*-
import os
from ncel.utils.document import *
import re
from ncel.utils.misc import loadWikiVocab

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
punction = '[{0}{1}{2}{3}]'.format(en_punctuation, zh_punctuation, en_sent_split_punc, zh_ssplit_punc)
ssplict_puncRE = re.compile('[{0}{1}]'.format(en_sent_split_punc, zh_ssplit_punc))

class XlwikiDataLoader():
    # genre indicates how difficult the mention is, 0 : easy, 1 : hard, 2 : all
    def __init__(self, path, genre=2, lowercase=False, wiki_map=None):
        self._fpath = path
        self.lowercase = lowercase
        self.is_hard = genre
        self._wiki_label2id, _ = wiki_map

    def _processLineSlice(self, line_slice, doc, sent):
        # split words in line slice
        # preprocess
        raw_tokens = re.split(punction, line_slice)

        for rt in raw_tokens:
            if len(rt) < 1: continue
            sent.append(rt)
            doc.tokens.append(rt)

        return sent

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
                en_wiki_name = re.sub(r'_', ' ', items[2])
                foreign_wiki_name = re.sub(r'_', ' ', items[3])
                if not isinstance(self._wiki_label2id, type(None)) and \
                                en_wiki_name not in self._wiki_label2id : continue
                en_wiki_id = self._wiki_label2id[en_wiki_name]
                # [_, _, _, new_start_offset, new_tokens_num, has_add_to_doc]
                tmp_mentions[j] = [en_wiki_name, en_wiki_id, foreign_wiki_name, -1, -1, False]
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
                            tmp_mentions[j][3] = len(doc.tokens)
                    sent = self._processLineSlice(line_slice, doc, sent)
                    # update mention end index
                    if p + base_offset in end_inx:
                        for j in end_inx[p + base_offset]:
                            if tmp_mentions[j][5]: continue
                            tmp_mentions[j][4] = len(doc.tokens)
                            if tmp_mentions[j][3] != -1:
                                m = Mention(doc, tmp_mentions[j][3], tmp_mentions[j][4],
                                            gold_ent_id=tmp_mentions[j][1], gold_ent_str=tmp_mentions[j][0])
                                m._gold_foreign_str = tmp_mentions[j][2]
                                doc.mentions.append(m)
                                tmp_mentions[j][5] = True

                    if p >= line_len: break
                    line_offset = p
                    split_inx_pos += 1
                if split_inx_pos == len(split_inx) and line_offset < line_len:
                    sent = self._processLineSlice(line[line_offset:], doc, sent)
                base_offset += line_len
                if len(sent) > 0:
                    doc.sentences.append(sent)
                sent = []
            for i, m in enumerate(doc.mentions):
                doc.mentions[i].setStrAndLength()
            yield doc

def load_data(text_path=None, mention_file=None, supplement=None,
              include_unresolved=False, lowercase=False, wiki_entity_file=None):
    assert not isinstance(text_path, type(None)), "xlwiki data requires raw path!"
    print("Loading", text_path)
    wiki_map = loadWikiVocab(wiki_entity_file)
    if supplement is None or supplement not in [0, 1, 2]: supplement=2
    docs = []
    doc_iter = XlwikiDataLoader(text_path, genre=supplement,
                                lowercase=lowercase, wiki_map=wiki_map)
    for doc in doc_iter.documents():
        docs.append(doc)
    return docs

if __name__ == "__main__":
    # Demo:
    docs = load_data(text_path='/home/caoyx/data/xlwikifier-wikidata/data/it/train/')
    print(docs[0].name)
    for m in docs[0].mentions:
        print("{0}, {1}, {2}, {3}".format(m._mention_start, m._mention_end, m._gold_ent_str, m._gold_foreign_str))
    print(docs[0].tokens)