# -*- coding: utf-8 -*-
import codecs
import re
import os
from ncel.utils.xmlProcessor import kbp15XmlHandler

def loadKbp2WikiMap(filename):
    q_map = {}
    # kbp_entity_id \t wiki
    with codecs.open(filename, 'r') as fin:
        for line in fin:
            items = re.split(r'\t', line.strip())
            if len(items) < 2: continue
            q_map[items[0]] = items[1]
    return q_map

class kbp15Formatter():
    def __init__(self, text_path, query_file, kbp2wiki_id_map, lang='en'):
        self._text_path = text_path
        self._query_file = query_file
        self._kbp2wiki_id_map = kbp2wiki_id_map
        self._lang = lang.upper()

    # return doc_mentions, {doc_id:[[ mention_text, wiki_id, start_offset, end_offset], ... ]}
    def getDocMentions(self):
        # load query to wiki map
        query_id_map = loadKbp2WikiMap(self._kbp2wiki_id_map)
        # load mentions
        doc_mentions = dict()
        with codecs.open(self._query_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                items = re.split(r'\t', line.strip())
                if len(items) < 5: continue
                tmp_items = re.split(r':|-', items[3])
                if len(tmp_items) < 3: continue

                doc_id = tmp_items[0]
                if not doc_id.startswith(self._lang) : continue
                start_p = int(tmp_items[1])
                end_p = int(tmp_items[2])
                mention_str = items[2]
                kbp_ent_id = items[4]
                if kbp_ent_id not in query_id_map : continue
                doc_mentions[doc_id] = doc_mentions.get(doc_id, [])
                doc_mentions[doc_id].append([mention_str, query_id_map[kbp_ent_id], start_p, end_p])
        # sort doc_mentions
        for id in doc_mentions:
            dm = doc_mentions[id]
            doc_mentions[id] = sorted(dm, key=lambda x:x[2])
        return doc_mentions

    def format(self, output_path):
        lang_doc_mentions = self.getDocMentions()
        # post
        files = os.listdir(self._text_path)
        postfix_inf = files[0].find(r'.')
        post_str = files[0][postfix_inf+1:]
        # process raw text
        new_all_mentions = dict()
        for doc_id in lang_doc_mentions:
            d_mentions = lang_doc_mentions[doc_id]
            fname = os.path.join(self._text_path, doc_id + '.' + post_str)
            if os.path.exists(fname):
                offset = 0
                tmp_lines = ''
                reader = kbp15XmlHandler(fname)
                for line in reader.originalText():
                    tmp_lines += line + ' '
                print(tmp_lines, d_mentions)
                for m in d_mentions:
                    print(tmp_lines[m[2]+39:m[3]+40])
            break



if __name__ == "__main__":
    kf = kbp15Formatter('/Users/ethan/Downloads/kbp15/eval/source_documents/eng/discussion_forum/',
                        '/Users/ethan/Downloads/kbp15/eval/tac_kbp_2015_tedl_evaluation_gold_standard_entity_mentions.tab',
                        '/Users/ethan/Downloads/kbp15/id.key')
    kf.format('./')