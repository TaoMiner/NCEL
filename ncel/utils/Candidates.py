# -*- coding: utf-8 -*-
import re
from ncel.utils.layers import cosSim
from ncel.utils.misc import loadWikiVocab

DEFAULT_PRIOR = 0.0

class CandidatesHandler:
    def __init__(self, file, vocab=None, lowercase=False):
        self._files = file.split(',')
        self._vocab = vocab
        self._mention_dict = None
        self._candidate_entities = None
        self._wikiid2label = None
        self._candidates_total = 0

        self._lowercase=lowercase
        # prior
        self._has_prior = False
        self._prior_mag = 100
        self._prior_mention_dict = None
        self._prior_entity_prob = None
        self._prior_mention_count = None


    def loadWikiid2Label(self, filename, id_vocab=None):
        _, self._wikiid2label = loadWikiVocab(filename, id_vocab=id_vocab)
        return self._wikiid2label

    def loadCandidates(self):
        for f in self._files:
            self.loadCandidatesFile(f)

    def loadCandidatesFile(self, filename):
        if isinstance(self._mention_dict, type(None)) : self._mention_dict = {}
        if isinstance(self._candidate_entities, type(None)) : self._candidate_entities = set()

        with open(filename, 'r', encoding='UTF-8') as fin:
            for line in fin:
                if self._lowercase : line = line.lower()
                items = re.split(r'\t', line.strip())
                ment_name = items[0]
                if len(items) < 2: continue
                # filter mention out of mention vocab
                if not isinstance(self._vocab, type(None)) and ment_name not in self._vocab: continue
                for c in items[1:]:
                    self._candidate_entities.add(c)
                    m_cand_set = self._mention_dict.get(ment_name, set())
                    m_cand_set.add(c)
                    self._mention_dict[ment_name] = m_cand_set
        self._candidates_total = sum([len(self._mention_dict[m]) for m in self._mention_dict])

    # load all candidates prior
    # enti_id \t gobal_prior \t cand_ment::=count \t ...
    def loadPrior(self, prior_file, mention_vocab=None, entity_vocab=None):
        if isinstance(self._prior_mention_dict, type(None)) : self._prior_mention_dict = {}
        if isinstance(self._prior_entity_prob, type(None)) : self._prior_entity_prob = {}
        if isinstance(self._prior_mention_count, type(None)): self._prior_mention_count = {}

        with open(prior_file, 'r', encoding='UTF-8') as fin:
            total_anchor_num = 0
            for line in fin:
                if self._lowercase: line = line.lower()
                items = re.split(r'\t', line.strip())
                entity_id = items[0]
                if entity_vocab is not None and entity_id not in entity_vocab : continue
                if len(items) < 3: continue
                ent_anchor_num = self._prior_entity_prob.get(entity_id, 0)
                for m_count_pair in items[2:]:
                    m_count = re.split(r'::=', m_count_pair)
                    if len(m_count) != 2 : continue
                    if mention_vocab is None or m_count[0] not in mention_vocab or\
                                    entity_id not in mention_vocab[m_count[0]]: continue
                    tmp_count = int(m_count[1])
                    ent_anchor_num += tmp_count
                    tmp_mention_ent = self._prior_mention_dict.get(m_count[0], {})
                    if entity_id in tmp_mention_ent:
                        tmp_mention_ent[entity_id] += tmp_count
                    else:
                        tmp_mention_ent[entity_id] = tmp_count
                    self._prior_mention_dict[m_count[0]] = tmp_mention_ent
                self._prior_entity_prob[entity_id] = float(ent_anchor_num)
                total_anchor_num += ent_anchor_num
        for ent in self._prior_entity_prob:
            self._prior_entity_prob[ent] = self._prior_entity_prob[ent] / total_anchor_num * self._prior_mag
        for m in self._prior_mention_dict:
            self._prior_mention_count[m] = sum([self._prior_mention_dict[m][k] for k in self._prior_mention_dict[m]])
        self._has_prior = True

    # order entity by prior
    def sort_candidates_by_prior(self, candidates):
        for i, cand in enumerate(candidates):
            m_count = self._prior_mention_count[cand.getMentionText()] if cand.getMentionText() in self._prior_mention_count else 0

            me_count = self._prior_mention_dict[cand.getMentionText()][cand.id] if cand.getMentionText() in self._prior_mention_dict \
                            and cand.id in self._prior_mention_dict[cand.getMentionText()] else 0
            pem = me_count / float(m_count) if m_count > 0 and me_count > 0 else DEFAULT_PRIOR

            candidates[i].setEntityMentionPrior(pem)
            candidates[i].setEntityPrior(self._prior_entity_prob.get(cand.id, DEFAULT_PRIOR))
        candidates = sorted(candidates, key=lambda x: x.getEntityMentionPrior())
        return candidates

    # return an ordered candidates list
    def get_candidates_for_mention(self, mention, vocab=None, is_eval=False, topn=0):
        if isinstance(self._mention_dict, type(None)) :
            self.loadCandidates()
        candidates = self._mention_dict.get(mention._mention_str, set())
        if not is_eval and mention.gold_ent_id() not in candidates:
            candidates.add(mention.gold_ent_id())
        # trim candidate sets by vocab
        candidates = [Candidate(mention, c) for c in candidates if vocab is not None and c in vocab]
        # sort by prior
        if self._has_prior:
            candidates = self.sort_candidates_by_prior(candidates)
        # crop by topn
        if topn > 0 and len(candidates) > topn:
            candidates = candidates[:topn]

        return candidates

    def add_candidates_to_mention(self, mention, vocab=None, is_eval=False, topn=0):
        mention.candidates = self.get_candidates_for_mention(mention,
                                   vocab=vocab, is_eval=is_eval, topn=topn)
        # set candidate labels
        is_NIL = True if isinstance(mention.gold_ent_id(), type(None)) else False
        is_trainable = False
        for i, cand in enumerate(mention.candidates):
            if not is_NIL and cand.id == mention.gold_ent_id():
                mention.candidates[i].setGold()
                is_trainable = True
        # NIL is only for inference
        if not is_eval and not is_trainable:
            mention._is_trainable = False

    def add_candidates_to_document(self, document, vocab=None, is_eval=False, topn=0):
        for i, mention in enumerate(document.mentions):
            self.add_candidates_to_mention(document.mentions[i],
                                           vocab=vocab, is_eval=is_eval, topn=topn)
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
        self._pe = DEFAULT_PRIOR
        self._pem = DEFAULT_PRIOR

        self._sense_context_sim = [DEFAULT_PRIOR] * 4
        self._mu_context_sim = [DEFAULT_PRIOR] * 4
        # contextual
        self._sense_emb = None
        self._sense_mu_emb = None
        self._context_emb = [None] * 4
        self._context_mu_emb = [None] * 4

    def getMentionText(self):
        return self._mention._mention_str

    def setContextSimilarity(self):
        if self._sense_emb is not None:
            for i in range(len(self._sense_context_sim)):
                if self._context_emb[i] is None : continue
                self._sense_context_sim[i] = cosSim(self._sense_emb, self._context_emb[i])
        if self._sense_mu_emb is not None:
            for i in range(len(self._mu_context_sim)):
                if self._context_mu_emb[i] is None : continue
                self._mu_context_sim[i] = cosSim(self._sense_mu_emb, self._context_mu_emb[i])

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

    def setEntityPrior(self, pe):
        self._pe = pe

    def setEntityMentionPrior(self, pem):
        self._pem = pem

    def getEntityPrior(self):
        return self._pe

    def getEntityMentionPrior(self):
        return self._pem

    def getSenseContextSim(self):
        return self._sense_context_sim[:2]

    def getMuContextSim(self):
        return self._mu_context_sim[:2]

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



# candidates is a list
def resortCandidates(candidates):
    candidates = sorted(candidates, key=lambda x:x.getEntityMentionPrior(), reverse=True)
    return candidates
