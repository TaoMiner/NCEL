try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import codecs
import re
import os
from ncel.utils.xmlProcessor import buildXml, kbp10XmlHandler

def loadKbp2WikiMap(filename):
    q_map = {}
    # kbp_entity_id \t wiki
    with codecs.open(filename, 'r') as fin:
        for line in fin:
            items = re.split(r'\t', line.strip())
            if len(items) < 2: continue
            q_map[items[0]] = items[1]
    return q_map

nonTextRE = re.compile(r'^<(.*)>$')
eleTagRE = re.compile(r'(<[^<>]+?>)([^<>]+?)(</[^<>]+?>)')
class kbp10Formatter():
    def __init__(self, text_path, query_xml, query_ans_file, kbp2wiki_id_map):
        self._text_path = text_path
        self._query_xml = query_xml
        self._query_ans_file = query_ans_file
        self._kbp2wiki_id_map = kbp2wiki_id_map

    # return doc_mentions, {doc_id:[[mention_text, wiki_id, [strat_offset]], ... ]}
    def getDocMentions(self):
        # load query to wiki map
        query_id_map = loadKbp2WikiMap(self._kbp2wiki_id_map)
        # load mentions
        q_xml = kbp10XmlHandler(self._query_xml)
        query_dict = dict()
        for (query_id, doc_id, mention_text) in q_xml.queries():
            query_dict[query_id] = [doc_id, mention_text, '', []]  # [_, _, wiki_id, start_offset]
        # query_id \t kbp_entity_id
        with codecs.open(self._query_ans_file, 'r') as fin:
            for line in fin:
                items = re.split(r'\t', line.strip())
                if len(items) < 2: continue
                if items[0] in query_dict and items[1] in query_id_map:
                    query_dict[items[0]][2] = query_id_map[items[1]]
        # rebuild mentions
        doc_mentions = dict()
        for qid in query_dict:
            if query_dict[qid][2] == '' and query_dict[qid][3] == '' : continue
            doc_id = query_dict[qid][0]
            doc_mentions[doc_id] = doc_mentions.get(doc_id, [])
            doc_mentions[doc_id].append(query_dict[qid][1:])
        return doc_mentions

    def format(self, output_path):
        all_doc_mentions = self.getDocMentions()
        # process raw text
        new_all_mentions = dict()
        for doc_id in all_doc_mentions:
            d_mentions = all_doc_mentions[doc_id]
            fname = os.path.join(self._text_path, doc_id + '.sgm')
            if os.path.exists(fname):
                offset = 0
                tmp_lines = ''
                with codecs.open(fname, 'r') as fin:
                    for line in fin:
                        line = line.strip()
                        if len(line) < 1: continue
                        m = nonTextRE.match(line)
                        if m is None:
                            for m in d_mentions:
                                tmp_idx = [m.start()+offset for m in re.finditer(m[0], line)]
                                m[-1].extend(tmp_idx)
                            offset += len(line)
                            tmp_lines += line+'\n'
                new_d_mentions = [[m[0], m[1], offset] for m in d_mentions for offset in m[2] if len(m[2]) > 0]
                if len(new_d_mentions) > 0:
                    new_d_mentions = sorted(new_d_mentions, key=lambda x: x[2])
                    new_all_mentions[doc_id] = new_d_mentions
                    out_fname = os.path.join(output_path, doc_id + '.txt')
                    with codecs.open(out_fname, 'w') as fout:
                        fout.write(tmp_lines+'\n')
        # format all doc mentions in xml
        buildXml(os.path.join(output_path, 'kbp2010.xml'), new_all_mentions)

if __name__ == "__main__":
    text_path = '/home/caoyx/data/kbp/kbp2010/TAC_2010_KBP_Source_Data/data/2010/wb/'
    query_xml = '/home/caoyx/data/kbp/kbp2010/TAC_2010_KBP_Evaluation_Entity_Linking_Gold_Standard_V1.0/data/tac_2010_kbp_evaluation_entity_linking_queries.xml'
    query_ans_file = '/home/caoyx/data/kbp/kbp2010/TAC_2010_KBP_Evaluation_Entity_Linking_Gold_Standard_V1.0/data/tac_2010_kbp_evaluation_entity_linking_query_types.tab'
    kbp2wiki_id_map = '/home/caoyx/data/kbp/kbp2010/id.key2010'
    kf = kbp10Formatter(text_path, query_xml, query_ans_file, kbp2wiki_id_map)
    kf.format('/home/caoyx/data/kbp/kbp_cl/kbp10/eval/')