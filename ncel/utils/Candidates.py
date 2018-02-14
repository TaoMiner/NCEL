# -*- coding: utf-8 -*-
import re
from ncel.utils.layers import cosSim

DEFAULT_PRIOR = 0.0

SOURCE = ['ppr','wiki_title', 'wiki_anchor', 'wiki_redirect', 'dictionary','yago','ncel']

class CandidatesHandler:
    def __init__(self, file, vocab=None, lowercase=False, id2label=None, label2id=None,
                 support_fuzzy=True, redirect_vocab=None):
        self._files = file.split(',')
        self._vocab = vocab         # mention vocab
        self._mention_dict = None       # {str:{ent:pem,...},...}
        self._entity_set = None     #entity set

        self._candidates_total = 0

        self._lowercase=lowercase
        self._id2label = id2label
        self._label2id = label2id
        self._support_fuzzy = support_fuzzy
        self._redirect_vocab = redirect_vocab

    def loadCandidates(self):
        for f in self._files:
            items = f.split(':')
            mention_dict, entity_set = self.loadCandidatesFromFile(items[0],items[1])
            self.addToCandidates(mention_dict, entity_set)
        self._candidates_total = sum([len(self._mention_dict[m]) for m in self._mention_dict])

    def addToCandidates(self, mention_dict, entity_set):
        if self._mention_dict is None:
            self._mention_dict = {}
        num_files = len(self._files)
        for m in mention_dict:
            tmp_cands = {}
            for c in mention_dict[m]:
                ent_id = c
                if self._redirect_vocab is not None and c in self._redirect_vocab:
                    ent_id = self._redirect_vocab[c]
                if ent_id in tmp_cands:
                    tmp_cands[ent_id] = (tmp_cands[ent_id] + mention_dict[m][c] / float(num_files)) / 2.0
                else : tmp_cands[ent_id] = mention_dict[m][c] / float(num_files)
            if m in self._mention_dict:
                for c in self._mention_dict[m]:
                    if c in tmp_cands:
                        tmp_cands[c] = (tmp_cands[c] + self._mention_dict[m][c]) / 2.0
                    else:
                        tmp_cands[c] = self._mention_dict[m][c]
            self._mention_dict[m] = tmp_cands

        if self._entity_set is None:
            self._entity_set = entity_set
        else:
            self._entity_set.update(entity_set)

    # SOURCE = ['ppr','wiki_title', 'wiki_anchor', 'wiki_redirect', 'dictionary','yago','ncel']
    def loadCandidatesFromFile(self, type, filename):
        mention_dict = {}
        entity_set = set()
        if type == SOURCE[0]:
            mention_dict, entity_set = self.loadCandidatesFromPPR(filename)
        elif type == SOURCE[1]:
            mention_dict, entity_set = self.loadCandidatesFromWikiTitle(filename)
        elif type == SOURCE[2]:
            mention_dict, entity_set = self.loadCandidatesFromWikiAnchor(filename)
        elif type == SOURCE[3]:
            mention_dict, entity_set = self.loadCandidatesFromWikiRedirect(filename)
        elif type == SOURCE[4]:
            mention_dict, entity_set = self.loadCandidatesFromDict(filename)
        elif type == SOURCE[5]:
            mention_dict, entity_set = self.loadCandidatesFromYago(filename)
        elif type == SOURCE[6]:
            mention_dict, entity_set = self.loadCandidatesFromNcel(filename)

        return mention_dict, entity_set

    # <string><tab><cprob><tab><id>
    def saveCandidatesToFile(self, filename):
        with open(filename, 'w', encoding='UTF-8') as fout:
            for m in self._mention_dict:
                for c in self._mention_dict[m]:
                    fout.write("{}\t{}\t{}\n".format(m, self._mention_dict[m][c], c))

    def candidateSoftmax(self, cand_dict):
        total_prior = sum([cand_dict[c] for c in cand_dict])
        if total_prior > 0:
            for c in cand_dict:
                if len(cand_dict) > 1:
                    cand_dict[c] = cand_dict[c] / total_prior
                else: cand_dict[c] = 1.0
        return cand_dict

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
                tmp_cand = mention_dict.get(m_str, {})
                tmp_cand[ent_id] = 1.0
                mention_dict[m_str] = tmp_cand
        for m in mention_dict:
            mention_dict[m] = self.candidateSoftmax(mention_dict[m])
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
                tmp_cand = mention_dict.get(m_str, {})
                tmp_cand[ent_id] = 1.0
                mention_dict[m_str] = tmp_cand

                # support fuzzy match, todo: for now, only support approximate person name (token=2)
                if self._support_fuzzy:
                    sf_items = re.split(r' ', m_str)
                    if len(sf_items) != 2: continue
                    for sf_m_str in sf_items:
                        if self._lowercase: sf_m_str = sf_m_str.lower()
                        # filter mention out of mention vocab
                        if not isinstance(self._vocab, type(None)) and sf_m_str not in self._vocab: continue
                        tmp_cand = mention_dict.get(sf_m_str, {})
                        tmp_cand[ent_id] = 1.0
                        mention_dict[sf_m_str] = tmp_cand
        for m in mention_dict:
            mention_dict[m] = self.candidateSoftmax(mention_dict[m])
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
                tmp_cand = mention_dict.get(ment_name, {})
                for c in items[1:]:
                    entity_set.add(c)
                    tmp_cand[c] = 1.0
                mention_dict[ment_name] = tmp_cand
        for m in mention_dict:
            mention_dict[m] = self.candidateSoftmax(mention_dict[m])
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
        for m in mention_dict:
            mention_dict[m] = self.candidateSoftmax(mention_dict[m])
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
        for m in mention_dict:
            mention_dict[m] = self.candidateSoftmax(mention_dict[m])
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
                tmp_cand = mention_dict.get(m_str, {})
                tmp_cand[ent_id] = 1.0
                mention_dict[m_str] = tmp_cand
        for m in mention_dict:
            mention_dict[m] = self.candidateSoftmax(mention_dict[m])
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
        for m in mention_dict:
            mention_dict[m] = self.candidateSoftmax(mention_dict[m])
        return mention_dict, entity_set

    # return an ordered candidates list
    def get_candidates_for_mention(self, mention, vocab=None, topn=0):
        assert self._mention_dict is not None, "load candidates first!"
        cand_dict = self._mention_dict.get(mention._mention_str, {})

        # trim candidate sets by vocab
        candidates = [Candidate(mention, c) for c in cand_dict if vocab is None or (vocab is not None and c in vocab)]
        for i, c in enumerate(candidates):
            candidates[i].setEntityMentionPrior(cand_dict[c.id])
        # sort by prior
        candidates = sorted(candidates, key=lambda x: x.getEntityMentionPrior(), reverse=True)
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

        self._mention = mention
        # feature
        self._is_gold = False
        # base
        self._pem = DEFAULT_PRIOR

        self._sense_context_sim = [DEFAULT_PRIOR] * 4
        self._mu_context_sim = [DEFAULT_PRIOR] * 4
        self._yamada_context_sim = [DEFAULT_PRIOR] * 2
        # contextual
        self._yamada_emb = None
        self._sense_emb = None
        self._sense_mu_emb = None
        self._context_emb = [None] * 4
        self._context_mu_emb = [None] * 4

    def getMentionText(self):
        return self._mention._mention_str

    def setContextSimilarity(self):
        if self._sense_emb is not None:
            for i in range(len(self._context_emb)):
                if self._context_emb[i] is None : continue
                self._sense_context_sim[i] = cosSim(self._sense_emb, self._context_emb[i])
        if self._sense_mu_emb is not None:
            for i in range(len(self._context_mu_emb)):
                if self._context_mu_emb[i] is None : continue
                self._mu_context_sim[i] = cosSim(self._sense_mu_emb, self._context_mu_emb[i])

    def setYamadaSimilarity(self):
        if self._yamada_emb is not None:
            for i in range(len(self._mention.context_emb)):
                if self._mention.context_emb[i] is None : continue
                self._yamada_context_sim[i] = cosSim(self._yamada_emb, self._mention.context_emb[i])

    def setSenseEmbeddings(self, emb):
        self._sense_emb = emb

    def setSenseMuEmbeddings(self, emb):
        self._sense_mu_emb = emb

    def setLeftContextEmbeddings(self, emb, is_mu):
        if is_mu:
            self._context_mu_emb[0] = emb
        else:
            self._context_emb[0] = emb

    def setRightContextEmbeddings(self, emb, is_mu):
        if is_mu:
            self._context_mu_emb[1] = emb
        else:
            self._context_emb[1] = emb

    def setLeftSentEmbeddings(self, emb, is_mu):
        if is_mu:
            self._context_mu_emb[2] = emb
        else:
            self._context_emb[2] = emb

    def setRightSentEmbeddings(self, emb, is_mu):
        if is_mu:
            self._context_mu_emb[3] = emb
        else:
            self._context_emb[3] = emb

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

    def getSenseContextSim(self):
        return self._sense_context_sim[:2]

    def getMuContextSim(self):
        return self._mu_context_sim[:2]

    def getYamadaContextSim(self):
        return self._yamada_context_sim[:2]

    def getSenseSentSim(self):
        return self._sense_context_sim[2:]

    def getMuSentSim(self):
        return self._mu_context_sim[2:]

    def getContextEmb(self):
        return self._context_emb[:2]

    def getSentEmb(self):
        return self._context_emb[2:]

    def getMuContextEmb(self):
        return self._context_mu_emb[:2]

    def getMuSentEmb(self):
        return self._context_mu_emb[2:]

    def getYamadaContextEmb(self):
        return self._mention.context_emb


NUM_PRIOR = 5
# candidates is a list
def resortCandidates(candidates, topn=NUM_PRIOR+2):
    new_candidates = candidates
    if topn > 0:
        topn_prior = NUM_PRIOR
        topn_sim = topn-NUM_PRIOR

        candidates = sorted(candidates, key=lambda x:x.getEntityMentionPrior(), reverse=True)
        new_candidates = candidates[:topn_prior]
        tmp_candidates = sorted(candidates[topn_prior:], key=lambda x: x._mu_context_sim[1], reverse=True)
        new_candidates += tmp_candidates[:topn_sim]
    return new_candidates
