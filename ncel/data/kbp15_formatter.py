from ncel.data.DataReader import *
from ncel.utils.xmlProcessor import buildXml

def getNlpToolUrl(lang):
    url = ''
    if lang == 'ENG':
        url = 'http://localhost:9001'
    elif lang == 'SPA':
        url = 'http://localhost:9002'
    elif lang == 'CMN':
        url = '/home/caoyx/data/dict.txt.big'
    return (lang, url)

class kbp15Formatter():
    # data_type = DATA_TYPE[0-2]
    def __init__(self, text_path, query_file, data_type, lang='eng'):
        self._text_path = text_path
        self._query_file = query_file
        self._lang = lang.upper()
        self._data_type = data_type

    def format(self, out_path, xml_file):
        dr = DataReader()
        dr.initNlpTool(getNlpToolUrl(self._lang))

        all_mentions = dr.loadKbpMentions(self._query_file)
        corpus = dr.readKbp(self._text_path, all_mentions, self._data_type)

        new_all_mentions = {}
        for doc in corpus:
            out_fname = os.path.join(out_path, doc.doc_id + '.txt')
            for m in doc.mentions:
                offset = len(' '.join(doc.text[:m[0]])) + 1
                length = len(' '.join(doc.text[m[0]:m[0] + m[1]]))
                assert length == len(m[3]), "wrong mention"
                new_all_mentions[doc.doc_id] = new_all_mentions.get(doc.doc_id, [])
                new_all_mentions[doc.doc_id].append([m[3], m[2], offset])
            with codecs.open(out_fname, 'w', encoding='utf-8') as fout:
                fout.write(' '.join(doc.text))

        # format all doc mentions in xml
        buildXml(xml_file, new_all_mentions)


if __name__ == "__main__":
    query_file = '/home/caoyx/data/kbp/LDC2017E03_TAC_KBP_Entity_Discovery_and_Linking_Comprehensive_Training_and_Evaluation_Data_2014-2016/data/2015/eval/tac_kbp_2015_tedl_evaluation_gold_standard_entity_mentions.tab'
    text_path = '/home/caoyx/data/kbp/LDC2017E03_TAC_KBP_Entity_Discovery_and_Linking_Comprehensive_Training_and_Evaluation_Data_2014-2016/data/2015/eval/source_documents/eng/discussion_forum/'
    kbp2wiki_id_map = '/home/caoyx/data/kbp/id.key2015'
    kf = kbp15Formatter(text_path, query_file, '2015', lang='eng')
    kf.format('/home/caoyx/data/kbp/kbp_cl/kbp15/eval/eng/df/', '/home/caoyx/data/kbp/kbp_cl/kbp15/eval/eng/kbp15_df.xml')