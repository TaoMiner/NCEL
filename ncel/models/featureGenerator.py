# -*- coding: utf-8 -*-
import numpy as np
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from ncel.utils.data import PADDING_ID

class FeatureGenerator:
    def __init__(self, initial_embeddings, embedding_dim,
                 str_sim=True, prior=True, hasAtt=True,
                 local_context_window=5, global_context_window=5):
        self._has_str_sim = str_sim
        self._has_prior = prior
        self._local_window = local_context_window
        self._global_window = global_context_window
        self._has_att = hasAtt
        (self.word_embeddings, self.entity_embeddings,
         self.sense_embeddings, self.mu_embeddings) = initial_embeddings
        self._dim = embedding_dim

        self._split_by_sent = True
        self._feature_dim = None

    def getFeatureDim(self):
        return self._feature_dim

    def getBaseFeature(self, doc):
        # (doc.n_candidates) * feature_num
        features = []
        for m in doc.mentions:
            # max prior
            max_pem = 0
            # number of candidates
            n_candidates = len(m.candidates)
            for c in m.candidates:
                if max_pem < c.getEntityMentionPrior():
                    max_pem = c.getEntityMentionPrior()

            features.extend(self.getMentionBaseFeature(m, n_candidates, max_pem))
        return features

    def getMentionBaseFeature(self, mention, n_candidates, max_pme):
        features = []
        for cand in mention.candidates:
            c_feature = self.getCandidateBaseFeature(cand, n_candidates, max_pme)
            features.append(c_feature)
        return features

    def getCandidateBaseFeature(self, candidate, num_candidates, max_prior):
        # base feature_num
        features = []
        m_label = candidate.mention
        # number of candidates
        features.append(num_candidates)
        # max_prior
        features.append(max_prior)

        # string similarity features
        if self._has_str_sim:
            c_label = candidate.str
            # mention_text_starts_or_ends_with_entity
            x = 1 if not isinstance(c_label, type(None)) and len(c_label) > 0 and (
                c_label.lower().startswith(m_label.lower()) or c_label.lower().endswith(
                    m_label.lower())) else 0
            features.append(x)
            # edit_distance
            features.append(
                normalized_damerau_levenshtein_distance(c_label.lower(), m_label.lower())
                if not isinstance(c_label, type(None)) and len(c_label) > 0 else 0)
        # prior
        if self._has_prior:
            # entity prior
            features.extend([candidate.getEntityPrior(), candidate.getEntityMentionPrior()])

        return features

    def buildContextFeature(self, doc):
        feature = []

        for mention in doc.mentions:
            lc_emb = None
            rc_emb = None
            ls_emb = None
            rs_emb = None
            if self._local_window >= 0:
                window = self._local_window if self._local_window>0 else None
                left_c = mention.left_context(max_len=window,
                                                split_by_sent=self._split_by_sent)
                right_c = mention.right_context(max_len=window,
                                                split_by_sent=self._split_by_sent)
                if len(left_c) > 0: lc_emb = self.getTokenEmbeds(left_c)
                if len(right_c) > 0: rc_emb = self.getTokenEmbeds(right_c)

            if self._global_window >= 0:
                window = self._global_window if self._global_window > 0 else None
                left_s = mention.left_sent(window)
                right_s = mention.right_sent(window)
                if len(left_s) > 0 : ls_emb = self.docEmbed(left_s)
                if len(right_s) > 0 : rs_emb = self.docEmbed(right_s)

            for cand in mention.candidates:
                tmp_f = []
                cand_emb = self.sense_embeddings[cand.id]
                cand_mu_emb = self.mu_embeddings[cand.id]
                if self._local_window >= 0:
                    left_sense_local_emb = self.getAttSentEmbed(cand_emb, lc_emb)
                    left_mu_local_emb = self.getAttSentEmbed(cand_mu_emb, lc_emb)
                    right_sense_local_emb = self.getAttSentEmbed(cand_emb, rc_emb)
                    right_mu_local_emb = self.getAttSentEmbed(cand_mu_emb, rc_emb)
                    tmp_f.extend([left_sense_local_emb, left_mu_local_emb,
                                  right_sense_local_emb, right_mu_local_emb])
                if self._global_window >= 0:
                    left_sense_global_emb = self.getAttSentEmbed(cand_emb, ls_emb)
                    left_mu_global_emb = self.getAttSentEmbed(cand_mu_emb, ls_emb)
                    right_sense_global_emb = self.getAttSentEmbed(cand_emb, rs_emb)
                    right_mu_global_emb = self.getAttSentEmbed(cand_mu_emb, rs_emb)
                    tmp_f.extend([left_sense_global_emb, left_mu_global_emb,
                                  right_sense_global_emb, right_mu_global_emb])
                feature.append(np.concatenate(tmp_f, axis=0))

        return np.array(feature)

    def getAttSentEmbed(self, query_emb, sent_embeds):
        if not isinstance(sent_embeds, type(None)) and sent_embeds.shape[0]>0:
            att = np.dot(sent_embeds, query_emb.transpose())
            embeds = np.dot(att.transpose(), sent_embeds)
        else:
            embeds = np.zeros(query_emb.shape[-1])
        return embeds

    def getTokenEmbeds(self, tokens):
        return self.word_embeddings.take(np.array(tokens).ravel(), axis=0)

    def sentEmbed(self, sent_tokens, weights=None):
        embeds = self.getTokenEmbeds(sent_tokens)
        embeds = np.reshape(embeds, (-1, self._dim))
        sent_embed = np.zeros(self._dim)
        if isinstance(weights, type(None)) or len(weights)!=len(sent_tokens):
            weights = np.ones(len(sent_tokens))
        for i,embed in enumerate(embeds):
            sent_embed += weights[i]*embed
        return sent_embed

    def docEmbed(self, sents):
        embeds = np.array([self.sentEmbed(x) for x in sents if len(x)>0])
        return embeds

    def getCandidateAndGoldIds(self, doc):
        candidate_ids = []
        gold_ids = []
        for m in doc.mentions:
            for i, c in enumerate(m.candidates):
                candidate_ids.append(c.id)
                gold_ids.append([1,0] if c.getIsGlod() else [0,1])
        return np.array(gold_ids), np.array(candidate_ids, dtype=np.int32)

    def getFeatures(self, doc):
        # doc.n_candidates * base_feature_dim
        base_feature = np.array(self.getBaseFeature(doc))
        assert base_feature.shape[0]==doc.n_candidates, "Error! No matched base feature extraction!"
        self._feature_dim = base_feature.shape[1]
        # doc.n_candidates * context_feature_dim
        if self._local_window >= 0 or self._global_window >= 0:
            context_feature = self.buildContextFeature(doc)
            assert context_feature.shape[0] == doc.n_candidates, "Error! No matched context feature extraction!"
            self._feature_dim += context_feature[0].shape[0]
            x = np.concatenate((base_feature, context_feature), axis=1)
        else:
            x = base_feature

        # candidate_ids: doc.n_candidates
        # gold_id: doc.n_candidates * 2
        y, candidate_ids = self.getCandidateAndGoldIds(doc)

        return x, candidate_ids, y
