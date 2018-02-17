# -*- coding: utf-8 -*-
import numba
import numpy as np
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from ncel.utils.layers import cosSim

class FeatureGenerator:
    def __init__(self, initial_embeddings, embedding_dim,
                 str_sim=True, prior=True, hasAtt=True,
                 local_context_window=5, global_context_window=5, use_embeddings=True):
        self._has_str_sim = str_sim
        self._has_prior = prior
        self._local_window = local_context_window
        self._global_window = global_context_window
        self._has_att = hasAtt
        (self.word_embeddings, self.entity_embeddings,
         self.sense_embeddings, self.mu_embeddings) = initial_embeddings
        self._dim = embedding_dim
        self._use_embeddings = use_embeddings


        self._split_by_sent = True
        if not self._split_by_sent and self._local_window == 0:
            self._split_by_sent = True
        self._feature_dim = None
        self.base_feature_dim = None

    # batch * tokens
    def padTokens(self, sents, PAD=0):
        max_len = max([len(s) for s in sents])
        return ([s+[PAD]*(max_len-len(s)) for s in sents])

    def AddEmbeddingFeatures(self, dataset):
        for i, doc in enumerate(dataset):
            for j, mention in enumerate(doc.mentions):
                if mention._is_trainable:
                    self.AddMentionEmbeddingFeatures(dataset[i].mentions[j])

    def AddMentionEmbeddingFeatures(self, mention):
        lc_emb = None
        rc_emb = None
        ls_emb = None
        rs_emb = None
        ylc_emb = None
        yrc_emb = None
        if self._local_window >= 0:
            window = self._local_window if self._local_window > 0 else None
            left_c = mention.left_context(max_len=window,
                                          split_by_sent=self._split_by_sent)
            right_c = mention.right_context(max_len=window,
                                            split_by_sent=self._split_by_sent)
            if len(left_c) > 0:
                lc_emb = self.getTokenEmbeds(left_c)
                if self._ntee_model is not None:
                    ylc_emb = self._ntee_model.get_text_vector(self._ntee_model.get_text_array(left_c))
            if len(right_c) > 0:
                rc_emb = self.getTokenEmbeds(right_c)
                if self._ntee_model is not None:
                    yrc_emb = self._ntee_model.get_text_vector(self._ntee_model.get_text_array(right_c))

        if self._global_window >= 0:
            window = self._global_window if self._global_window > 0 else None
            left_s = mention.left_sent(window)
            right_s = mention.right_sent(window)
            if len(left_s) > 0:
                ls_emb = self.docEmbed(left_s)
            if len(right_s) > 0:
                rs_emb = self.docEmbed(right_s)
        mention.setContextEmb(ylc_emb, yrc_emb)
        for i, candidate in enumerate(mention.candidates):
            self.AddCandidateEmbeddingFeatures(mention.candidates[i], lc_emb, rc_emb, ls_emb, rs_emb)
            mention.candidates[i].setContextSimilarity()
            if self._ntee_model is not None:
                mention.candidates[i]._yamada_emb = self._ntee_model.get_entity_vector(candidate.id)
                mention.candidates[i].setYamadaSimilarity()

    def AddCandidateEmbeddingFeatures(self, candidate, left_context_embeddings, right_context_embeddings,
                             left_sent_embeddings, right_sent_embeddings):
        cand_emb = self.sense_embeddings[candidate.id]
        candidate.setSenseEmbeddings(cand_emb)

        candidate.setLeftContextEmbeddings(self.getFeatureEmbeddings(cand_emb, left_context_embeddings), False)
        candidate.setRightContextEmbeddings(self.getFeatureEmbeddings(cand_emb, right_context_embeddings), False)
        candidate.setLeftSentEmbeddings(self.getFeatureEmbeddings(cand_emb, left_sent_embeddings), False)
        candidate.setRightSentEmbeddings(self.getFeatureEmbeddings(cand_emb, right_sent_embeddings), False)

        if self._use_mu:
            cand_mu_emb = self.mu_embeddings[candidate.id]
            candidate.setSenseMuEmbeddings(cand_mu_emb)
            candidate.setLeftContextEmbeddings(self.getFeatureEmbeddings(cand_emb, left_context_embeddings), True)
            candidate.setRightContextEmbeddings(self.getFeatureEmbeddings(cand_emb, right_context_embeddings), True)
            candidate.setLeftSentEmbeddings(self.getFeatureEmbeddings(cand_emb, left_sent_embeddings), True)
            candidate.setRightSentEmbeddings(self.getFeatureEmbeddings(cand_emb, right_sent_embeddings), True)

    def getFeatureEmbeddings(self, query_emb, embeddings):
        if embeddings is not None:
            lc_emb = self.getSeqEmbeddings(embeddings, query_emb=query_emb) if \
                self._has_att else self.getSeqEmbeddings(embeddings)
        else:
            lc_emb = np.zeros(self._dim)
        return lc_emb

    def getFeatureDim(self):
        return self._feature_dim

    def setBaseFeature(self, doc):
        for m in doc.mentions:
            # max prior
            max_pem = 0
            # number of candidates
            n_candidates = len(m.candidates)
            for c in m.candidates:
                if max_pem < c.getEntityMentionPrior():
                    max_pem = c.getEntityMentionPrior()
            for cand in m.candidates:
                c_feature = self.getCandidateBaseFeature(cand, n_candidates, max_pem)
                cand.setBaseFeature(np.array(c_feature, dtype=float))
                if self.base_feature_dim is None:
                    self.base_feature_dim = len(c_feature)

    def getCandidateBaseFeature(self, candidate, num_candidates, max_prior):
        # base feature_num
        features = []
        m_label = candidate.getMentionText()
        # number of candidates
        features.append(num_candidates)
        # max_prior
        features.append(max_prior)

        # string similarity features
        if self._has_str_sim:
            c_label = candidate.label
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
            features.append(candidate.getEntityMentionPrior())

        return features

    @numba.autojit
    def getSeqEmbeddings(self, sent_embeds, query_emb=None):
        if query_emb is None:
            embeds = np.mean(sent_embeds, axis=0)
        else:
            att = np.dot(query_emb, sent_embeds.transpose())
            embeds = np.dot(att, sent_embeds)
        return embeds

    def getTokenEmbeds(self, tokens):
        return self.word_embeddings.take(np.array(tokens).ravel(), axis=0)

    def docEmbed(self, sents):
        sents_embed = []
        for s in sents:
            if len(s) > 0:
                embeds_array = self.getTokenEmbeds(s)
                embeds_array = np.reshape(embeds_array, (-1, self._dim))
                embeds = self.getSeqEmbeddings(embeds_array)
                sents_embed.append(embeds)
        doc_embeds = None
        if len(sents_embed) > 0:
            doc_embeds = np.array(sents_embed)
        return doc_embeds

    def getCandidateAndGoldIds(self, doc):
        # cand_id: mention_index
        candidate_ids = []
        gold_ids = []
        for i, m in enumerate(doc.mentions):
            for j, c in enumerate(m.candidates):
                candidate_ids.append([c.id, i])
                gold_ids.append(1 if c.getIsGlod() else 0)
        return np.array(gold_ids), np.array(candidate_ids)

    # todo: yamada embedding may be empty
    def getContextFeature(self, doc):
        features = []
        for m in doc.mentions:
            for c in m.candidates:
                tmp_f = []
                if self._local_window >= 0:
                    tmp_f.extend(c.getSenseContextSim())
                    if self._ntee_model is not None:
                        tmp_f.extend(c.getYamadaContextSim())
                    if self._use_mu:
                        tmp_f.extend(c.getMuContextSim())
                if self._global_window >= 0:
                    tmp_f.extend(c.getSenseSentSim())
                    if self._use_mu:
                        tmp_f.extend(c.getMuSentSim())
                if self._use_embeddings :
                    tmp_context_emb = np.concatenate(c.getContextEmb(), axis=0)
                    tmp_f = np.concatenate((np.array(tmp_f), c._sense_emb, tmp_context_emb), axis=0)
                features.append(tmp_f)
        return np.array(features)

    def getFeatures(self, doc):
        # doc.n_candidates * base_feature_dim
        base_feature = np.array(self.getBaseFeature(doc))
        if len(base_feature.shape) > 1:
            assert base_feature.shape[0]==doc.n_candidates, "Error! No matched base feature extraction!"
            self._feature_dim = base_feature.shape[1]
        else:
            assert doc.n_candidates==1, "Error! No matched base feature extraction!"
            self._feature_dim = base_feature.shape[0]
        # doc.n_candidates * context_feature_dim
        if self._local_window >= 0 or self._global_window >= 0:
            context_feature = self.getContextFeature(doc)
            if len(context_feature.shape) > 1:
                assert context_feature.shape[0] == doc.n_candidates, "Error! No matched base feature extraction!"
                self._feature_dim += context_feature.shape[1]
            else:
                assert doc.n_candidates == 1, "Error! No matched base feature extraction!"
                self._feature_dim += context_feature.shape[0]

            x = np.concatenate((base_feature, context_feature), axis=1)
        else:
            x = base_feature

        # candidate_ids: doc.n_candidates
        # gold_id: doc.n_candidates
        y, candidate_ids = self.getCandidateAndGoldIds(doc)

        return x, candidate_ids, y
