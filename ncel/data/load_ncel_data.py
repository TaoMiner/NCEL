# -*- coding: utf-8 -*-
from ncel.utils.document import *
from ncel.utils.xmlProcessor import *
from ncel.utils.misc import loadWikiVocab
import os
import re

MAX_DOCS = 50000

def _NcelFileToDocIterator(fpath):
    files = os.listdir(fpath)
    for fname in files:
        postfix_inf = fname.rfind(r'.')
        doc_name = fname if postfix_inf == -1 else fname[:postfix_inf]
        f = open(os.path.join(fpath, fname), 'r')
        lines = f.readlines()
        yield (doc_name, lines)

en_punctuation = " \'\",:()\-\n"
zh_punctuation = " ·＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"
en_sent_split_punc = "\.?!"
zh_ssplit_punc = "！？｡。"
punction = '[{0}{1}]'.format(en_punctuation, zh_punctuation)
ssplict_puncRE = re.compile('[{0}{1}]'.format(en_sent_split_punc, zh_ssplit_punc))

class NcelDataLoader(xmlHandler):
    def __init__(self, rawtext_path, mention_fname, include_unresolved=False, lowercase=False,
                 wiki_map=None):
        super(NcelDataLoader, self).__init__(['mention', 'wikiID'], ['offset', 'length'])
        self._fpath = rawtext_path
        self._m_fname = mention_fname
        self._include_unresolved = include_unresolved
        self.lowercase = lowercase
        _, self._wiki_id2label = wiki_map

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
                if len(sent) > 0:
                    doc.sentences.append(sent)
                    sent = []
                if len(rt) > 1 :
                    doc.tokens.append(rt[dot_idx + 1:])
                    sent.append(rt[dot_idx+1:])
            elif dot_idx == len(rt)-1:
                doc.tokens.append(rt[:dot_idx])
                sent.append(rt[:dot_idx])
                doc.sentences.append(sent)
                sent = []
            else:
                doc.tokens.append(rt[:dot_idx])
                doc.tokens.append(rt[dot_idx+1:])
                sent.append(rt[:dot_idx])
                doc.sentences.append(sent)
                sent = []
                sent.append(rt[dot_idx + 1:])
        return sent

    def getNormDocName(self, doc_name):
        doc_name = doc_name.replace(' ', '_').replace('?', '_').replace('*', '_').replace('/',
                      '_').replace('\\', '_').replace('>', '_').replace('<', '_').replace(':',
                      '_').replace('"', '_').replace('|', '_')
        return doc_name

    def documents(self):
        all_mentions = dict()
        for (doc_name, mentions) in self.process(self._m_fname):
            postfix_inf = doc_name.rfind(r'.')
            doc_name = doc_name if postfix_inf == -1 else doc_name[:postfix_inf]
            doc_name = self.getNormDocName(doc_name)
            all_mentions[doc_name] = list(mentions)
        i=0
        for (doc_name, doc_lines) in _NcelFileToDocIterator(self._fpath):
            if doc_name not in all_mentions : continue
            if i > MAX_DOCS : break
            # create doc mention offset index list
            start_inx = dict()
            end_inx = dict()
            tmp_mentions = dict()
            split_inx = set()
            doc_mentions = all_mentions[doc_name]
            for j, doc_mention in enumerate(doc_mentions):

                wiki_label = self._wiki_id2label.get(doc_mention['wikiID'], '')

                doc_start_inx = doc_mention['offset']
                doc_end_inx = doc_mention['offset'] + doc_mention['length']
                split_inx.add(doc_start_inx)
                split_inx.add(doc_end_inx)
                start_inx[doc_start_inx] = start_inx.get(doc_start_inx, [])
                start_inx[doc_start_inx].append(j)
                end_inx[doc_end_inx] = end_inx.get(doc_end_inx, [])
                end_inx[doc_end_inx].append(j)
                # [_, _, new_start_offset, new_tokens_num, has_add_to_doc]

                tmp_mentions[j] = [doc_mention['mention'], wiki_label, doc_mention['wikiID'], -1, -1, False]

            # skip those don't have any mention
            if len(tmp_mentions) < 1: continue
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
                            tmp_mentions[j][3] = len(doc.tokens)
                    sent = self._processLineSlice(line_slice, doc, sent)
                    # update mention end index
                    if p + base_offset in end_inx:
                        for j in end_inx[p + base_offset]:
                            if tmp_mentions[j][5]: continue
                            tmp_mentions[j][4] = len(doc.tokens)
                            if tmp_mentions[j][3] == -1 : continue
                            if tmp_mentions[j][2] == 'NIL':
                                m = Mention(doc, tmp_mentions[j][3], tmp_mentions[j][4], is_NIL=True)
                            else:
                                m = Mention(doc, tmp_mentions[j][3], tmp_mentions[j][4],
                                        gold_ent_id=tmp_mentions[j][2], gold_ent_str=tmp_mentions[j][1])
                            doc.mentions.append(m)
                            tmp_mentions[j][5] = True

                    if p >= line_len : break
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

    def mentions(self):
        for doc in self.documents():
            for mention in doc.mentions:
                yield mention

def load_data(text_path=None, mention_file=None, kbp_id2wikiid_file=None,
              genre=0, include_unresolved=False, lowercase=False,
              wiki_entity_file=None):
    assert not isinstance(text_path, type(None)) and not isinstance(mention_file, type(None)),\
        "Ncel data requires raw text path and mention file!"
    print("Loading {0}, {1}".format(text_path,mention_file))
    wiki_map = loadWikiVocab(wiki_entity_file)
    docs = []
    doc_iter = NcelDataLoader(text_path, mention_file,
                              include_unresolved=include_unresolved, lowercase=lowercase,
                              wiki_map=wiki_map)
    for doc in doc_iter.documents():
        docs.append(doc)
    return docs

if __name__ == "__main__":
    # Demo:
    docs = load_data(text_path='/home/caoyx/data/WNED/wned-datasets/ace2004/RawText/',
                     mention_file='/home/caoyx/data/WNED/wned-datasets/ace2004/ace2004.xml')
    print(docs[0].doc_name)
    print(docs[0].mentions)
    print(docs[0].tokens)