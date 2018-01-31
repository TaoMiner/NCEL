# -*- coding: utf-8 -*-
import re
from ncel.utils.data import loadWikiVocab

DEFAULT_PRIOR = 0.0


class CandidatesHandler:
    def __init__(self, file, vocab=None, lowercase=False):
        self._files = file.split(',')
        self._vocab = vocab
        self._mention_dict = None
        self._candidate_entities = None
        self._lowercase=lowercase
        self._prior_mag = 100

        self._mention_count = None
        self._wikiid2label = None

        self._candidates_total = 0

    def loadWikiid2Label(self, filename, id_vocab=None):
        _, self._wikiid2label = loadWikiVocab(filename, id_vocab=id_vocab)
        return self._wikiid2label

    def loadCandidates(self):
        for f in self._files:
            self.loadCandidatesFile(f)

    def loadCandidatesFile(self, filename):
        if isinstance(self._mention_dict, type(None)) : self._mention_dict = {}
        if isinstance(self._candidate_entities, type(None)) : self._candidate_entities = {}

        with open(filename, 'r', encoding='UTF-8') as fin:
            for line in fin:
                if self._lowercase : line = line.lower()
                items = re.split(r'\t', line.strip())
                ment_name = items[0]
                if len(items) < 2: continue
                # filter mention out of mention vocab
                if not isinstance(self._vocab, type(None)) and ment_name not in self._vocab: continue
                tmp_cand = self._mention_dict.get(ment_name, {})
                for c in items[1:]:
                    if c not in self._candidate_entities : self._candidate_entities[c] = 0
                    if c not in tmp_cand :
                        tmp_cand[c] = 0
                        self._candidates_total += 1
                self._mention_dict[ment_name] = tmp_cand

    # enti_id \t gobal_prior \t cand_ment::=count \t ...
    def loadPrior(self, prior_file):
        if isinstance(self._mention_dict, type(None)) :
            self.loadCandidates()
        with open(prior_file, 'r', encoding='UTF-8') as fin:
            total_anchor_num = 0
            for line in fin:
                if self._lowercase: line = line.lower()
                items = re.split(r'\t', line.strip())
                if len(items) < 3 or items[0] not in self._candidate_entities: continue

                ent_anchor_num = self._candidate_entities[items[0]]
                for mention in items[2:]:
                    tmp_items = re.split(r'::=', mention)
                    if len(tmp_items) != 2 or tmp_items[0] not in self._mention_dict: continue
                    tmp_count = int(tmp_items[1])
                    ent_anchor_num += tmp_count
                    if items[0] in self._mention_dict[tmp_items[0]]:
                        self._mention_dict[tmp_items[0]][items[0]] += tmp_count
                self._candidate_entities[items[0]] = float(ent_anchor_num)
                total_anchor_num += ent_anchor_num
        for ent in self._candidate_entities:
            self._candidate_entities[ent] = self._candidate_entities[ent] / total_anchor_num * self._prior_mag
        self._mention_count = {}
        for m in self._mention_dict:
            self._mention_count[m] = sum([self._mention_dict[m][k] for k in self._mention_dict[m]])

    # return an ordered candidates list
    def get_candidates_for_mention(self, mention_str, vocab=None, topn=None):
        if isinstance(self._mention_dict, type(None)) :
            self.loadCandidates()
        candidates = self._mention_dict.get(mention_str, {})
        # trim candidate sets by vocab and topn
        tmp_candidates = []
        for cand in candidates:
            if not isinstance(vocab, type(None)) and cand not in vocab : continue
            wiki_label = None
            if not isinstance(self._wikiid2label, type(None)) and\
                            cand in self._wikiid2label:
                wiki_label = self._wikiid2label[cand]
            pem = float(candidates[cand])/self._mention_count[mention_str] if mention_str in self._mention_count and \
                self._mention_count[mention_str] > 0 else DEFAULT_PRIOR
            #
            tmp_c = Candidate(mention_str, cand, str=wiki_label)
            tmp_c.setEntityPrior(self._candidate_entities.get(cand, DEFAULT_PRIOR))
            tmp_c.setEntityMentionPrior(pem)
            tmp_candidates.append(tmp_c)
        if not isinstance(topn, type(None)) and len(tmp_candidates) > topn:
            tmp_candidates = sorted(tmp_candidates, key=lambda x:x.getEntityMentionPrior())[:topn]

        return tmp_candidates

    def add_candidates_to_mention(self, mention, vocab=None, topn=None):
        mention.candidates = self.get_candidates_for_mention(mention._mention_str, vocab=vocab, topn=topn)
        # set candidate labels
        is_NIL = True if isinstance(mention.gold_ent_id(), type(None)) else False
        for i, cand in enumerate(mention.candidates):
            if not is_NIL and cand.id == mention.gold_ent_id():
                mention.candidates[i].setGold()

    def add_candidates_to_document(self, document, vocab=None, topn=None):
        for i, mention in enumerate(document.mentions):
            self.add_candidates_to_mention(document.mentions[i], vocab=vocab, topn=topn)
            # compute document candidats number
            document.n_candidates += len(document.mentions[i].candidates)


def getCandidateHandler():
    return CandidatesHandler

class Candidate():
    def __init__(self, mention_str, id, str=None):
        self.id = id
        self.str = str
        self.mention = mention_str

        self._is_gold = False

        self._pe = DEFAULT_PRIOR
        self._pem = DEFAULT_PRIOR

    def setGold(self):
        self._is_gold = True

    def getIsGlod(self):
        return self._is_gold

    def setEntityPrior(self, pe):
        self._pe = pe

    def setEntityMentionPrior(self, pem):
        self._pem = pem

    def getEntityPrior(self):
        return self._pe

    def getEntityMentionPrior(self):
        return self._pem
