import codecs
import regex as re
import string
import os
from pycorenlp import StanfordCoreNLP
import json as simplejson
import jieba
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

jieba.set_dictionary('/home/caoyx/data/dict.txt.big')

xmlDefRE = re.compile(r'<?xml.*?>')
textHeadRE = re.compile(r'<TEXT>|<HEADLINE>')
textTailRE = re.compile(r'</TEXT>|</HEADLINE>')
sourceRE = re.compile(r'<SOURCE>.*</SOURCE>')
timeRE = re.compile(r'<DATE_TIME>.*</DATE_TIME>')
nonTextRE = re.compile(r'^<[^<>]+?>$')
eleTagRE = re.compile(r'(<[^<>]+?>)([^<>]+?)(</[^<>]+?>)')
propTagRE = re.compile(r'(<[^<>]+?/>)')
puncRE = re.compile("[{0}]".format(re.escape(string.punctuation)))
zh_punctuation = "！？｡。·＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
zhpunc = re.compile("[{0}]".format(re.escape(zh_punctuation)))


class Doc:
    def __init__(self):
        self.doc_id = -1
        self.text = []          # [w, ..., w] token lists
        self.mentions = []      # [[w_index, mention_lenth, ent_id, ment_str], ...]

class DataReader:
    def __init__(self):
        self.lang = ''
        self.nlp = None
        self.prop = None

    def loadKbidMap(self, filename):
        id_map = {}
        with codecs.open(filename, 'r') as fin:
            for line in fin:
                items = re.split(r'\t', line.strip())
                if len(items) < 2: continue
                id_map[items[0]] = items[1]
        return id_map

    def initNlpTool(self, input):
        (lang, url) = input
        self.lang = lang
        if lang != 'CMN':
            self.nlp = StanfordCoreNLP(url)
            self.prop = {'annotators': 'tokenize, ssplit, lemma', 'outputFormat': 'json'}
        elif os.path.isfile(url):
            jieba.set_dictionary(url)
            self.nlp = jieba
        print("set nlp tool!")

    def tokenize(self, sent):
        if isinstance(self.nlp, type(None)):
            print("please init nlp tool!")
            return
        tokens = []
        if self.lang == 'CMN':
            tokens = self.nlp.tokenize(sent)
        else:
            results = self.nlp.annotate(sent, properties=self.prop)
            for sent in results['sentences']:
                for token in sent['tokens']:
                    tokens.append([token['word'], token['characterOffsetBegin'], token['characterOffsetEnd'], token['lemma']])
        return tokens

    # {doc_id:[[startP, endP, wikiId, mention_str],...], ...}
    def loadKbpMentions(self, file):
        # load mentions
        doc_mentions = dict()
        with codecs.open(file, 'r', encoding='utf-8') as fin:
            for line in fin:
                items = re.split(r'\t', line.strip())
                if len(items) < 5: continue
                tmp_items = re.split(r':|-', items[3])
                if len(tmp_items) < 3: continue

                doc_id = tmp_items[0]
                # remove other lang docs
                if not doc_id.startswith(self.lang): continue
                start_p = int(tmp_items[1])
                end_p = int(tmp_items[2])
                mention_str = items[2]

                kbp_ent_id = items[4]

                doc_mentions[doc_id] = doc_mentions.get(doc_id, [])
                doc_mentions[doc_id].append([start_p, end_p, kbp_ent_id, mention_str])
        # sort doc_mentions
        for id in doc_mentions:
            dm = doc_mentions[id]
            doc_mentions[id] = sorted(dm, key=lambda x: x[0])
        return doc_mentions

    def readKbp(self, text_path, mentions, data_type):
        files = os.listdir(text_path)
        corpus = []
        if data_type.startswith('2015'):
            extract = self.extractKBP15Text
        elif data_type.endswith('df'):
            extract = self.extractKBP16DfText
        else :
            extract = self.extractKBP16NwText
        for f in files:
            postfix_inf = f.find(r'.')
            if postfix_inf == -1 : continue
            filename = f[:postfix_inf]
            if filename in mentions:
                sents = extract(os.path.join(text_path, f))
                doc = self.readDoc(sents, mentions[filename])
                doc.doc_id = filename
                corpus.append(doc)
        return corpus

    # return original text and its count, according to dataset year
    def extractKBP15Text(self, file):
        sents = []
        tree = ET.ElementTree(file=file)
        for seg_e in tree.iterfind('DOC/TEXT/SEG'):
            cur_pos = int(seg_e.attrib['start_char'])
            for text_e in seg_e.iter(tag='ORIGINAL_TEXT'):
                line = text_e.text
                source_m = sourceRE.match(line.strip())
                time_m = timeRE.match(line.strip())
                m = nonTextRE.match(line.strip())
                if m == None and len(line.strip()) > 0 and source_m == None and time_m == None:
                    sents.append([cur_pos-40, text_e.text])     # ignore the begining xml definition
        return sents
    # skip all the lines <...>
    def extractKBP16DfText(self, file):
        sents = []
        cur_pos = -1
        with codecs.open(file, 'r') as fin:
            for line in fin:
                cur_len = len(line)
                m = nonTextRE.match(line.strip())
                if m == None and len(line.strip()) > 0:
                    sents.append([cur_pos, line])
                cur_pos += cur_len
        return sents
    # sents: [[sent,start_pos, sent_line]]
    def extractKBP16NwText(self, file):
        sents = []
        isDoc = False
        cur_pos = -1
        with codecs.open(file, 'r') as fin:
            for line in fin:
                cur_len = len(line)
                if isDoc:
                    # text ends or <P>
                    text_m = nonTextRE.match(line.strip())
                    tail_m = textTailRE.match(line.strip())
                    if text_m != None or tail_m != None:
                        cur_pos += cur_len
                        if tail_m != None : isDoc = False
                        continue
                    if len(line.strip()) > 0:
                        sents.append([cur_pos, line])
                else:
                    head_m = textHeadRE.match(line.strip())
                    # text starts
                    if head_m != None:
                        isDoc = True
                cur_pos += cur_len
        return sents
    # return class doc
    def readDoc(self, sents, mentions):
        doc = Doc()
        mention_index = 0
        tmp_map = {}
        for sent in sents:
            cur_pos = sent[0]
            line = sent[1]
            # some line contains <>..</>   or  <.../>
            tag_pos = []
            tag_len = 0
            tag_index = 0
            for etm in eleTagRE.finditer(line):
                tag_pos.append(etm.span(1))
                tag_pos.append(etm.span(3))
            for ptm in propTagRE.finditer(line):
                tag_pos.append(ptm.span(1))
            tmp_line = line
            if len(tag_pos) > 0:
                tag_pos = sorted(tag_pos, key=lambda x:x[0])
                tmp_line = line[0:tag_pos[0][0]]
                for i in range(len(tag_pos)-1):
                    tmp_line += line[tag_pos[i][1]:tag_pos[i+1][0]]
                tmp_line += line[tag_pos[len(tag_pos)-1][1]:]
            if len(tmp_line.strip()) < 1: continue
            lspace = len(tmp_line) - len(tmp_line.lstrip())
            tokens = self.tokenize(tmp_line)
            # tokens : [[word, start, end, lemma],...]  no lemma for chinese
            for token in tokens:
                w = token[0]
                lemma = w if self.lang == 'CMN' else token[3]
                t_start = cur_pos + token[1] + 1 + lspace
                t_end = cur_pos + token[2] + 1 + lspace
                if tag_index < len(tag_pos) and t_start + tag_len >= tag_pos[tag_index][0] + cur_pos +1:
                    tag_len += tag_pos[tag_index][1] - tag_pos[tag_index][0]
                    tag_index += 1
                if len(tag_pos) > 0:
                    t_start += tag_len
                    t_end += tag_len
                tmp_seg = [[0, -1, 0], [len(w), 1000, 1]]
                # put all the mention boundary into the set
                for j in range(mention_index, len(mentions),1):
                    if mentions[j][0] > t_end-1 : break
                    if mentions[j][0] >= t_start and mentions[j][0] < t_end:
                        tmp_seg.append([mentions[j][0]-t_start, j, 0])
                    if mentions[j][1] >= t_start and mentions[j][1] < t_end:
                        tmp_seg.append([mentions[j][1]-t_start+1, j, 1])
                if len(tmp_seg) <= 2 :       # if no mention is in this token
                    doc.text.append(lemma)
                else:
                    tmp_seg = sorted(tmp_seg, key=lambda x:(x[0], x[2], x[1]))
                    for j in range(len(tmp_seg)-1):
                        m_index = tmp_seg[j][1]
                        add_text = 1
                        if tmp_seg[j][0] == 0 and tmp_seg[j+1][0] == len(w) :
                            doc.text.append(lemma)
                        elif tmp_seg[j+1][0] > tmp_seg[j][0]:
                            doc.text.append(w[tmp_seg[j][0]:tmp_seg[j+1][0]])
                        else:
                            add_text = 0
                        if m_index == -1 or m_index >= 1000: continue
                        if tmp_seg[j][2] == 0:
                            tmp_map[m_index] = len(doc.text)-add_text
                        elif m_index in tmp_map:
                            doc.mentions.append([tmp_map[m_index], len(doc.text)-tmp_map[m_index]-add_text, mentions[m_index][2], ' '.join(doc.text[tmp_map[m_index]:tmp_map[m_index]+len(doc.text)-tmp_map[m_index]-add_text])])
        return doc

if __name__ == '__main__':
    dr = DataReader()