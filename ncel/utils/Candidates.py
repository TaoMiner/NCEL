# -*- coding: utf-8 -*-
import re
from ncel.utils.layers import cosSim

DEFAULT_PRIOR = 0.0

# ppr, wiki_title, wiki_redirect, yago, combine, uniform
# wiki_anchor, dictionary, average
SOURCE = ['ppr','wiki_title', 'wiki_anchor', 'wiki_redirect', 'dictionary','yago','ncel']

class CandidatesHandler:
    def __init__(self, file, vocab=None, lowercase=False, id2label=None, label2id=None,
                 support_fuzzy=True, redirect_vocab=None, topn=0):
        self._files = file.split(',')
        self._vocab = vocab         # mention vocab
        self._mention_dict = None       # {str:{ent:pem,...},...}
        self._entity_set = None     #entity set

        self._uni_mention_dict = None  # {str:{ent,...},...}

        self._candidates_total = 0

        self._topn = topn

        self._lowercase=lowercase
        self._id2label = id2label
        self._label2id = label2id
        self._support_fuzzy = support_fuzzy
        self._redirect_vocab = redirect_vocab

    def loadCandidates(self):
        for f in self._files:
            items = f.split(':')
            is_uniform, mention_dict, entity_set = self.loadCandidatesFromFile(items[0],items[1])
            if is_uniform:
                self.addToUniformCandidates(mention_dict)
            else:
                self.addToCandidates(mention_dict)
        self.combineCandidates()
        self._candidates_total = sum([len(self._mention_dict[m]) for m in self._mention_dict])

    def combineCandidates(self):
        if self._uni_mention_dict is not None:
            for m in self._uni_mention_dict:
                # add a uniform distribution to all candidates in mention dict
                uni_prior = 1.0/float(len(self._uni_mention_dict[m]))
                tmp_cands = self._mention_dict.get(m, {})
                for c in self._uni_mention_dict[m]:
                    if c not in tmp_cands:
                        tmp_cands[c] = uni_prior
                    else: tmp_cands[c] = max(uni_prior, tmp_cands[c])
                self._mention_dict[m] = tmp_cands
        if self._entity_set is None:
            self._entity_set = set()
        for m in self._mention_dict:
            # sort
            sort_cand = sorted(self._mention_dict[m].items(), key=lambda x: x[1], reverse=True)
            if self._topn > 0:
                sort_cand = sort_cand[:self._topn]
            self._mention_dict[m] = self.candidateSoftmax(sort_cand)
            self._entity_set.update([cand[0] for cand in sort_cand])

    def addToUniformCandidates(self, mention_dict):
        if self._uni_mention_dict is None:
            self._uni_mention_dict = {}
        for m in mention_dict:
            tmp_cand_set = self._uni_mention_dict.get(m, set())
            # replace candidate id if is redirect
            for c in mention_dict[m]:
                ent_id = c
                if self._redirect_vocab is not None and c in self._redirect_vocab:
                    ent_id = self._redirect_vocab[c]
                tmp_cand_set.add(ent_id)
            self._uni_mention_dict[m] = tmp_cand_set

    def addToCandidates(self, mention_dict):
        if self._mention_dict is None:
            self._mention_dict = {}
        for m in mention_dict:
            tmp_cands = self._mention_dict.get(m, {})
            # replace candidate id if is redirect
            for c in mention_dict[m]:
                ent_id = c
                if self._redirect_vocab is not None and c in self._redirect_vocab:
                    ent_id = self._redirect_vocab[c]

                if ent_id in tmp_cands:
                    # todo: average or max(better)
                    tmp_cands[ent_id] = max(tmp_cands[ent_id], mention_dict[m][c])
                else : tmp_cands[ent_id] = mention_dict[m][c]
            self._mention_dict[m] = tmp_cands

    # SOURCE = ['ppr','wiki_title', 'wiki_anchor', 'wiki_redirect', 'dictionary','yago','ncel']
    def loadCandidatesFromFile(self, type, filename):
        mention_dict = {}
        entity_set = set()
        is_uniform = False
        if type == SOURCE[0]:
            mention_dict, entity_set = self.loadCandidatesFromPPR(filename)
            is_uniform = True
        elif type == SOURCE[1]:
            mention_dict, entity_set = self.loadCandidatesFromWikiTitle(filename)
            is_uniform = True
        elif type == SOURCE[2]:
            mention_dict, entity_set = self.loadCandidatesFromWikiAnchor(filename)
        elif type == SOURCE[3]:
            mention_dict, entity_set = self.loadCandidatesFromWikiRedirect(filename)
            is_uniform = True
        elif type == SOURCE[4]:
            mention_dict, entity_set = self.loadCandidatesFromDict(filename)
        elif type == SOURCE[5]:
            mention_dict, entity_set = self.loadCandidatesFromYago(filename)
            is_uniform = True
        elif type == SOURCE[6]:
            mention_dict, entity_set = self.loadCandidatesFromNcel(filename)

        return is_uniform, mention_dict, entity_set

    # <string><tab><cprob><tab><id>
    def saveCandidatesToFile(self, filename):
        with open(filename, 'w', encoding='UTF-8') as fout:
            for m in self._mention_dict:
                for c in self._mention_dict[m]:
                    fout.write("{}\t{}\t{}\n".format(m, c[1], c[0]))

    def candidateSoftmax(self, cand_list):
        total_prior = sum([cand[1] for cand in cand_list])
        new_cands = []
        if total_prior > 0:
            for i, cand in enumerate(cand_list):
                new_cands.append([cand[0], cand[1]/total_prior if len(cand_list) > 1 else 1.0])
        return new_cands

    # wiki redirect
    def loadCandidatesFromWikiRedirect(self, filename):
        mention_dict = {}
        entity_set = set()
        with open(filename, 'r', encoding='UTF-8') as fin:
            for line in fin:
                items = re.split(r'\t', line.strip())
                m_str = items[0]
                if self._lowercase: m_str = m_str.lower()
                if len(items) < 2 or (not isinstance(self._vocab, type(None)) and m_str not in self._vocab) : continue
                ent_id = items[1]
                tmp_cand_set = mention_dict.get(m_str, set())
                tmp_cand_set.add(ent_id)
                mention_dict[m_str] = tmp_cand_set
                entity_set.add(ent_id)
        return mention_dict, entity_set

    # wiki title
    # omit brackets
    def loadCandidatesFromWikiTitle(self, filename):
        mention_dict = {}
        entity_set = set()
        bracketRE = re.compile(r'\(.*\)')
        with open(filename, 'r', encoding='UTF-8') as fin:
            for line in fin:
                items = re.split(r'\t', line.strip())
                m_str = bracketRE.sub('', items[0]).strip()
                if self._lowercase: m_str = m_str.lower()
                if len(items) < 2 or (not isinstance(self._vocab, type(None)) and m_str not in self._vocab): continue
                ent_id = items[1]
                entity_set.add(ent_id)
                tmp_cand_set = mention_dict.get(m_str, set())
                tmp_cand_set.add(ent_id)
                mention_dict[m_str] = tmp_cand_set

                # support fuzzy match, todo: for now, only support approximate person name (token=2)
                if self._support_fuzzy:
                    sf_items = re.split(r' ', m_str)
                    if len(sf_items) != 2: continue
                    for sf_m_str in sf_items:
                        if self._lowercase: sf_m_str = sf_m_str.lower()
                        # filter mention out of mention vocab
                        if not isinstance(self._vocab, type(None)) and sf_m_str not in self._vocab: continue
                        tmp_cand_set = mention_dict.get(sf_m_str, set())
                        tmp_cand_set.add(ent_id)
                        mention_dict[sf_m_str] = tmp_cand_set
        return mention_dict, entity_set

    # Personalized Page Rank for Named Entity Disambiguation
    # str \t id \t ...
    def loadCandidatesFromPPR(self, filename):
        mention_dict = {}
        entity_set = set()
        with open(filename, 'r', encoding='UTF-8') as fin:
            for line in fin:
                if self._lowercase : line = line.lower()
                items = re.split(r'\t', line.strip())
                ment_name = items[0]
                if len(items) < 2: continue
                # filter mention out of mention vocab
                if not isinstance(self._vocab, type(None)) and ment_name not in self._vocab: continue
                tmp_cand_set = mention_dict.get(ment_name, set())
                tmp_cand_set.update(items[1:])
                entity_set.update(items[1:])
                mention_dict[ment_name] = tmp_cand_set
        return mention_dict, entity_set

    # anchors
    # enti_id \t gobal_prior \t cand_ment::=count \t ...
    def loadCandidatesFromWikiAnchor(self, filename):
        mention_dict = {}
        entity_set = set()
        with open(filename, 'r', encoding='UTF-8') as fin:
            for line in fin:
                if self._lowercase: line = line.lower()
                items = re.split(r'\t', line.strip())
                if len(items) < 3 : continue
                ent_id = items[0]
                for ment_count in items[2:]:
                    mc = ment_count.split('::=')
                    if len(mc) != 2: continue
                    m_str = mc[0]
                    count = float(mc[1])
                    if not isinstance(self._vocab, type(None)) and m_str not in self._vocab: continue
                    tmp_cand = mention_dict.get(m_str, {})
                    tmp_count = tmp_cand.get(ent_id, 0.0)
                    tmp_cand[ent_id] = tmp_count + count
                    mention_dict[m_str] = tmp_cand
                    entity_set.add(ent_id)
        return mention_dict, entity_set

    # A cross-lingual dictionary for english wikipedia con- cepts
    # <string><tab><cprob><space><url>[<space><score>]*
    def loadCandidatesFromDict(self, filename):
        mention_dict = {}
        entity_set = set()
        assert self._label2id is not None, "Dict needs label2id dict!"
        with open(filename, 'r', encoding='UTF-8', errors='ignore') as fin:
            for line in fin:
                items = re.split(r'\t', line.strip())
                if len(items) < 2 : continue
                m_str = items[0]
                if self._lowercase: m_str = m_str.lower()
                if not isinstance(self._vocab, type(None)) and m_str not in self._vocab: continue
                tmp_items = items[1].split(' ')
                if len(tmp_items) < 2 : continue
                cprob = float(tmp_items[0])
                wiki_label = re.sub(r'_', ' ', tmp_items[1])
                if wiki_label not in self._label2id : continue
                ent_id = self._label2id[wiki_label]
                entity_set.add(ent_id)
                tmp_cand = mention_dict.get(m_str, {})
                if ent_id in tmp_cand:
                    tmp_cand[ent_id] = (tmp_cand[ent_id] + cprob ) / 2.0
                else:
                    tmp_cand[ent_id] = cprob
                mention_dict[m_str] = tmp_cand
        return mention_dict, entity_set

    # <Alexander_de_Brus,_Earl_of_Carrick> \t rdfs:label \t "Alexander de Brus, Earl of Carrick"@eng .
    # <wordnet_superfecta_100507539> \t skos:prefLabel \t "superfecta"@eng .
    # <Edward_Berkowitz> \t <redirectedFrom> \t "Edward D. Berkowitz"@eng .
    def loadCandidatesFromYago(self, filename):
        mention_dict = {}
        entity_set = set()
        assert self._label2id is not None, "Yago needs label2id dict!"
        with open(filename, 'r', encoding='UTF-8') as fin:
            for line in fin:
                line = line.strip()
                items = re.split(r'\t', line.strip())
                if len(items) < 3 or items[1] not in ['rdfs:label', 'skos:prefLabel',
                      '<redirectedFrom>'] or not items[2].endswith('@eng .'): continue
                m_str = items[2][:-6].strip('"')
                if self._lowercase: m_str = m_str.lower()
                if not isinstance(self._vocab, type(None)) and m_str not in self._vocab: continue
                wiki_label = re.sub(r'_', ' ', items[0].strip('<>'))
                if wiki_label not in self._label2id: continue
                ent_id = self._label2id[wiki_label]
                entity_set.add(ent_id)
                tmp_cand_set = mention_dict.get(m_str, set())
                tmp_cand_set.add(ent_id)
                mention_dict[m_str] = tmp_cand_set
        return mention_dict, entity_set

    # <string><tab><cprob><tab><id>
    def loadCandidatesFromNcel(self, filename):
        mention_dict = {}
        entity_set = set()
        with open(filename, 'r', encoding='UTF-8') as fin:
            for line in fin:
                items = re.split(r'\t', line.strip())
                if len(items) < 3: continue
                m_str = items[0]
                if self._lowercase: m_str = m_str.lower()
                if not isinstance(self._vocab, type(None)) and m_str not in self._vocab: continue
                cprob = float(items[1])
                ent_id = items[2]
                entity_set.add(ent_id)
                tmp_cand = mention_dict.get(m_str, {})
                tmp_cand[ent_id] = cprob
                mention_dict[m_str] = tmp_cand
        return mention_dict, entity_set

    # return an ordered candidates list
    def get_candidates_for_mention(self, mention, vocab=None, topn=0):
        assert self._mention_dict is not None, "load candidates first!"
        cand_list = self._mention_dict.get(mention._mention_str, [])

        candidates = []
        # trim candidate sets by vocab
        for i, cand in enumerate(cand_list):
            if vocab is None or (vocab is not None and cand[0] in vocab):
                c = Candidate(mention, cand[0])
                c.setEntityMentionPrior(cand[1])
                if self._id2label is not None:
                    c.setLabel(self._id2label[cand[0]] if cand[0] in self._id2label else '')
                candidates.append(c)
        # crop by topn
        if topn > 0 and len(candidates) > topn:
            candidates = candidates[:topn]
        return candidates

    def add_candidates_to_mention(self, mention, vocab=None, topn=0):
        mention.candidates = self.get_candidates_for_mention(mention, vocab=vocab, topn=topn)

    def add_candidates_to_document(self, document, vocab=None, topn=0):
        for i, mention in enumerate(document.mentions):
            self.add_candidates_to_mention(document.mentions[i], vocab=vocab, topn=topn)
            # compute document candidats number
            document.n_candidates += len(document.mentions[i].candidates)


def getCandidateHandler():
    return CandidatesHandler

class Candidate():
    def __init__(self, mention, id, label=None):
        self.id = id
        self.label = label

        self._sense_id = None

        self._mention = mention
        # feature
        self._is_gold = False
        # base
        self._pem = DEFAULT_PRIOR
        self._base = None

    def setLabel(self, label):
        self.label = label

    def setBaseFeature(self, f):
        self._base = f

    def getBaseFeature(self):
        return self._base

    def setSense(self, id):
        self._sense_id = id

    def getSense(self):
        return self._sense_id

    def getMentionText(self):
        return self._mention._mention_str

    def getMention(self):
        return self._mention

    def setGold(self):
        self._is_gold = True

    def getIsGlod(self):
        return self._is_gold

    def setEntityMentionPrior(self, pem):
        self._pem = pem

    def getEntityMentionPrior(self):
        return self._pem
