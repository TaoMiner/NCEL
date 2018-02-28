# -*- coding: utf-8 -*-
import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from ncel.utils.layers import Embed, to_gpu, SubGraphConvolution, LayerNormalization

DEFAULT_SIM = 0.0

def build_model(base_feature_dim, initial_embeddings, FLAGS):
    model_cls = PNCEL
    layers_dim = [2000]
    use_contexts2 = FLAGS.use_lr_context
    use_att = FLAGS.att
    use_embedding_feature = True
    neighbor_window = 3
    bias = True
    return model_cls(
        base_feature_dim,
        initial_embeddings,
        layers_dim=layers_dim,
        bias=bias,
        gc_ln=FLAGS.mlp_ln,
        dropout=FLAGS.dropout,
        fine_tune_loaded_embeddings=FLAGS.fine_tune_loaded_embeddings,
        use_contexts2=use_contexts2,
        use_att=use_att,
        use_embedding_feature=use_embedding_feature,
        neighbor_window=neighbor_window
    )

class PNCEL(nn.Module):
    def __init__(self,
                 base_dim,
                 initial_embeddings,
                 layers_dim=[],
                 bias=True,
                 gc_ln=False,
                 dropout=0.0,
                 fine_tune_loaded_embeddings=None,
                 use_contexts2=True,
                 use_att=True,
                 use_embedding_feature=True,
                 neighbor_window=3
                 ):
        super(PNCEL, self).__init__()

        self._use_contexts2 = use_contexts2
        self._use_att = use_att
        self._dropout_rate = dropout
        self._neighbor_window = neighbor_window
        self._gc_ln = gc_ln

        word_embeddings, entity_embeddings, sense_embeddings, mu_embeddings = initial_embeddings
        word_vocab_size, word_embedding_dim = word_embeddings.shape
        entity_vocab_size, entity_embedding_dim = entity_embeddings.shape
        self._has_sense = True if sense_embeddings is not None else False
        if self._has_sense:
            sense_vocab_size, sense_embedding_dim = sense_embeddings.shape
        self._dim = word_embedding_dim
        assert self._dim == entity_embedding_dim and not (
        self._has_sense and self._dim != sense_embedding_dim), "unmatched dim!"

        self.word_embed = Embed(self._dim, word_vocab_size,
                                vectors=word_embeddings, fine_tune=fine_tune_loaded_embeddings)

        self.entity_embed = Embed(self._dim, entity_vocab_size,
                                  vectors=entity_embeddings, fine_tune=fine_tune_loaded_embeddings)

        if self._has_sense:
            self.sense_embed = Embed(self._dim, sense_vocab_size,
                                     vectors=sense_embeddings, fine_tune=fine_tune_loaded_embeddings)
            self.mu_embed = Embed(self._dim, sense_vocab_size,
                                  vectors=mu_embeddings, fine_tune=fine_tune_loaded_embeddings)

        self.embeds = [self.word_embed, self.entity_embed, self.sense_embed, self.mu_embed]

        self.cos = nn.CosineSimilarity(dim=2, eps=1e-8)

        #
        self._use_embedding_feature = use_embedding_feature
        self._feature_dim = base_dim + 4 * 3

        if self._use_embedding_feature:
            emb_num = 3 if self._use_contexts2 else 2
            self._feature_dim += emb_num * self._dim

        self._num_layers = len(layers_dim)
        features_dim = self._feature_dim

        if self._gc_ln:
            self.ln_inp = LayerNormalization(self._feature_dim)

        for i in range(self._num_layers):
            hidden_dim = layers_dim[i]
            setattr(self, 'l{}'.format(i), SubGraphConvolution(features_dim, hidden_dim, bias=bias))
            setattr(self, 'd{}'.format(i), hidden_dim)
            features_dim = hidden_dim
        self.gc_classifier = SubGraphConvolution(features_dim, 1, bias=bias)

    # types: index of [word,entity,sense,mu]
    def run_embed(self, x, type):
        embeds = self.embeds[type](x)
        embeds = F.dropout(embeds, self._dropout_rate, training=self.training)
        return embeds

    def getEmbFeatures(self, sents, q_emb=None):
        batch_size, cand_num, seq_length = sents.size()

        sents_emb = self.run_embed(sents, 0)
        sents_emb = sents_emb.view(batch_size * cand_num, seq_length, -1)

        if q_emb is not None:
            att = torch.bmm(q_emb.unsqueeze(1), sents_emb.transpose(1, 2))
            f_emb = torch.bmm(att, sents_emb).squeeze()
        else:
            f_emb = torch.mean(sents_emb, dim=1)

        return f_emb

    def leftNeighbor(self, emb, mv_steps, margin_col, mask):
        left_neigh_emb = torch.cat([margin_col, emb[:-mv_steps, :]], dim=0)
        left_neigh_emb = left_neigh_emb * mask
        return left_neigh_emb

    def rightNeighbor(self, emb, mv_steps, margin_col, mask):
        right_neigh_emb = torch.cat([emb[mv_steps:, :], margin_col], dim=0)
        right_neigh_emb = right_neigh_emb * mask
        return right_neigh_emb

    def getNeighborMask(self, num_mentions, dim):
        batch_size, cand_num = num_mentions.shape
        # batch * cand_num
        margin_col = to_gpu(Variable(torch.zeros(1, cand_num), requires_grad=False))

        right_mask = to_gpu(Variable(torch.from_numpy(num_mentions), requires_grad=False)).float()
        left_mask = torch.cat([margin_col, right_mask[:-1, :]], dim=0)

        # (batch * cand_num) * dim
        right_mask_expand = right_mask.view(-1).unsqueeze(1).expand(batch_size * cand_num, dim)
        left_mask_expand = left_mask.view(-1).unsqueeze(1).expand(batch_size * cand_num, dim)
        return left_mask_expand, right_mask_expand

    def getNeighborMentionEmbeddingsForCandidate(self, mention_emb, margin_col, cand_num, neighbor_window, left_mask, right_mask):
        # left_neighs: (batch_size*cand_num) * window * dim
        tmp_left_neigh_list = []
        tmp_left_neigh_list.append(self.leftNeighbor(mention_emb, cand_num,
                                                             margin_col, left_mask))
        for i in range(neighbor_window - 1):
            tmp_left_neigh_list.append(self.leftNeighbor(tmp_left_neigh_list[i],
                                              cand_num, margin_col, left_mask))
        for i, neigh in enumerate(tmp_left_neigh_list):
            tmp_left_neigh_list[i] = tmp_left_neigh_list[i].unsqueeze(1)
        left_neighs = torch.cat(tmp_left_neigh_list, dim=1)

        tmp_right_neigh_list = []
        tmp_right_neigh_list.append(self.rightNeighbor(mention_emb, cand_num,
                                                               margin_col, right_mask))
        for i in range(neighbor_window - 1):
            tmp_right_neigh_list.append(self.rightNeighbor(tmp_right_neigh_list[i],
                                                    cand_num, margin_col, right_mask))
        for i, neigh in enumerate(tmp_right_neigh_list):
            tmp_right_neigh_list[i] = tmp_right_neigh_list[i].unsqueeze(1)
        right_neighs = torch.cat(tmp_right_neigh_list, dim=1)
        # neigh_emb: (batch_size*cand_num) * 2window * dim
        neigh_emb = torch.cat((left_neighs, right_neighs), dim=1)
        # neigh_emb: (batch_size*cand_num) * dim
        neigh_emb = torch.mean(neigh_emb, dim=1)
        return neigh_emb

    # emb : (batch * cand_num) * dim
    # mask: (batch * cand_num) * dim
    def getNeighborMentionEmbeddings(self, mention_embeddings, neighbor_window, num_mentions):
        batch_size, cand_num = num_mentions.shape
        entity_emb, sense_emb, mu_emb = mention_embeddings
        _, dim = entity_emb.size()
        left_mask, right_mask = self.getNeighborMask(num_mentions, dim)
        margin_col = to_gpu(Variable(torch.zeros(cand_num, dim), requires_grad=False))

        neibor_ment_entity_emb = self.getNeighborMentionEmbeddingsForCandidate(entity_emb,
                                 margin_col, cand_num, neighbor_window, left_mask, right_mask)
        neibor_ment_sense_emb = None
        neibor_ment_mu_emb = None
        if sense_emb is not None:
            neibor_ment_sense_emb = self.getNeighborMentionEmbeddingsForCandidate(sense_emb,
                                margin_col, cand_num, neighbor_window, left_mask, right_mask)
        if mu_emb is not None:
            neibor_ment_mu_emb = self.getNeighborMentionEmbeddingsForCandidate(mu_emb,
                                margin_col, cand_num, neighbor_window, left_mask, right_mask)
        return neibor_ment_entity_emb, neibor_ment_sense_emb, neibor_ment_mu_emb

    def getExpandNeighCandidates(self, neigh_emb, batch_size, cand_num, dim):
        tmp_neigh_cand = neigh_emb.view(batch_size, cand_num, dim)
        tmp_neigh_cand = tmp_neigh_cand.unsqueeze(1).expand(batch_size, cand_num,
                                                            cand_num, dim)
        # (batch * cand) * cand * dim
        tmp_neigh_cand = tmp_neigh_cand.contiguous().view(batch_size * cand_num, cand_num, dim)
        return tmp_neigh_cand

    def getNeighCandidates(self, emb, window, num_mentions):
        batch_size, cand_num = num_mentions.shape
        _, dim = emb.size()
        left_mask, right_mask = self.getNeighborMask(num_mentions, dim)
        margin_col = to_gpu(Variable(torch.zeros(cand_num, dim), requires_grad=False))
        left_list = []
        # (batch * cand) * dim
        tmp_neigh = self.leftNeighbor(emb, cand_num, margin_col, left_mask)
        tmp_neigh_cand = self.getExpandNeighCandidates(tmp_neigh, batch_size, cand_num, dim)
        left_list.append(tmp_neigh_cand)
        for i in range(window - 1):
            tmp_neigh = self.leftNeighbor(emb, cand_num, margin_col, left_mask)
            tmp_neigh_cand = self.getExpandNeighCandidates(tmp_neigh, batch_size,
                                                           cand_num, dim)
            left_list.append(tmp_neigh_cand)
        # (batch * cand) * (window*cand) * dim
        left_cands = torch.cat(left_list, dim=1)

        right_list = []
        tmp_neigh = self.rightNeighbor(emb, cand_num, margin_col, right_mask)
        tmp_neigh_cand = self.getExpandNeighCandidates(tmp_neigh, batch_size, cand_num, dim)
        right_list.append(tmp_neigh_cand)
        for i in range(window - 1):
            tmp_neigh = self.rightNeighbor(emb, cand_num, margin_col, right_mask)
            tmp_neigh_cand = self.getExpandNeighCandidates(tmp_neigh, batch_size,
                                                           cand_num, dim)
            right_list.append(tmp_neigh_cand)
        # (batch * cand) * (window*cand) * dim
        right_cands = torch.cat(right_list, dim=1)

        # (batch * cand) * (cand_num*window*2) * dim
        neigh_cands = torch.cat((left_cands, right_cands), dim=1)
        return neigh_cands

    # cand_emb: (batch * cand) * dim
    # adj: (batch_size * cand_num) * (2*window*cand_num+1)
    def buildGraph(self, cand_emb, window, num_mentions, thred=0):
        batch_size, cand_num = num_mentions.shape
        # (batch * cand) * (cand_num*window*2) * dim
        neigh_cands = self.getNeighCandidates(cand_emb, window, num_mentions)

        # (batch * cand) * (cand_num*window*2) * dim
        cand_emb_expand = cand_emb.unsqueeze(1).expand(batch_size * cand_num,
                                                       2*window*cand_num, self._dim)
        # (batch * cand) * (cand_num*window*2)
        adj = self.cos(cand_emb_expand, neigh_cands)
        # add self connection
        margin_col = to_gpu(Variable(torch.ones(batch_size*cand_num, 1),
                                     requires_grad=False))
        # size: (batch * cand) * (cand_num*window*2+1)
        adj = torch.cat((adj, margin_col), dim=1)
        # normalize
        adj = torch.clamp(adj, thred, 1)
        adj = F.normalize(adj, p=1, dim=1)
        return adj

    # f: (batch * cand) * dim
    # gf_neighbors: (batch * cand) * (cand_num*window*2+1) * dim
    def getExpandFeature(self, f, window, num_mentions):
        _, f_dim = f.size()
        # (batch * cand) * (cand_num*window*2) * dim
        gf_neighbors = self.getNeighCandidates(f, window, num_mentions)
        # (batch * cand) * (cand_num*window*2+1) * dim
        gf_neighbors = torch.cat((gf_neighbors, f.unsqueeze(1)), dim=1)
        return gf_neighbors


    def getCandidateEmbedding(self, candidates, candidates_sense=None):
        candidates = to_gpu(Variable(torch.from_numpy(candidates))).long()
        cand_entity_emb = self.run_embed(candidates, 1)
        cand_sense_emb = None
        cand_mu_emb = None
        if candidates_sense is not None and self._has_sense:
            candidates_sense = to_gpu(Variable(torch.from_numpy(candidates_sense))).long()
            cand_sense_emb = self.run_embed(candidates_sense, 2)
            cand_mu_emb = self.run_embed(candidates_sense, 3)
        return cand_entity_emb, cand_sense_emb, cand_mu_emb

    def getTokenEmbedding(self, tokens, candidate_embeddings=None):
        tokens = to_gpu(Variable(torch.from_numpy(tokens))).long()

        sense_emb = None
        mu_emb = None
        if candidate_embeddings is not None:
            cand_entity_emb, cand_sense_emb, cand_mu_emb = candidate_embeddings
            entity_emb = self.getEmbFeatures(tokens, q_emb=cand_entity_emb)
            if cand_sense_emb is not None:
                sense_emb = self.getEmbFeatures(tokens, q_emb=cand_sense_emb)
            if cand_mu_emb is not None:
                mu_emb = self.getEmbFeatures(tokens, q_emb=cand_mu_emb)
        else:
            entity_emb = self.getEmbFeatures(tokens)

        return entity_emb, sense_emb, mu_emb

    def getCandidateSimilarity(self, embeddings, candidate_embeddings):
        cand_entity_emb, cand_sense_emb, cand_mu_emb = candidate_embeddings
        entity_emb, sense_emb, mu_emb = embeddings

        cand_entity_emb_expand = cand_entity_emb.unsqueeze(1)
        sim1 = torch.bmm(cand_entity_emb_expand, entity_emb.unsqueeze(2)).squeeze(2)

        sim2 = DEFAULT_SIM
        sim3 = DEFAULT_SIM
        if cand_sense_emb is not None and cand_mu_emb is not None:
            cand_sense_emb_expand = cand_sense_emb.unsqueeze(1)
            cand_mu_emb_expand = cand_mu_emb.unsqueeze(1)
            sim2 = torch.bmm(cand_sense_emb_expand, sense_emb.unsqueeze(2)).squeeze(2)
            sim3 = torch.bmm(cand_mu_emb_expand, mu_emb.unsqueeze(2)).squeeze(2)

        return sim1, sim2, sim3

    # contexts1 : batch * candidate * tokens
    # contexts2 : batch * candidate * tokens
    # base_feature : batch * candidate * features, numpy
    # candidates : batch * candidate
    # candidates_entity: batch * candidate
    # length: batch
    # num_mentions: batch * cand
    def forward(self, contexts1, base_feature, candidates, mention_tokens,
                contexts2=None, candidates_sense=None, num_mentions=None, length=None):
        batch_size, cand_num, _ = base_feature.shape
        features = []
        # to gpu
        base_feature = to_gpu(Variable(torch.from_numpy(base_feature))).float()
        base_feature = base_feature.view(batch_size * cand_num, -1)
        features.append(base_feature)

        # candidate mask
        length_mask = None
        if length is not None:
            lengths_var = to_gpu(Variable(torch.from_numpy(length), requires_grad=False)).long()
            # batch_size * cand_num
            length_mask = sequence_mask(lengths_var, cand_num).float()

        # get emb, (batch * cand) * dim
        candidate_embeddings = self.getCandidateEmbedding(candidates, candidates_sense)
        cand_emb1, cand_emb2, cand_emb3 = candidate_embeddings

        # get context emb
        context1_emb = self.getTokenEmbedding(contexts1,
                       candidate_embeddings=candidate_embeddings if self._use_att else None)

        # get contextual similarity, (batch * cand) * contextual_sim
        con1_sims = self.getCandidateSimilarity(context1_emb, candidate_embeddings)
        features.extend(con1_sims)

        con2_sims = DEFAULT_SIM, DEFAULT_SIM, DEFAULT_SIM
        con2_emb_cand1 = None
        if self._use_contexts2 and contexts2 is not None:
            context2_emb = self.getTokenEmbedding(contexts2,
                        candidate_embeddings=candidate_embeddings if self._use_att else None)
            con2_emb_cand1, con2_emb_cand2, con2_emb_cand3 = context2_emb
            # get contextual similarity, (batch * cand) * contextual_sim
            con2_sims = self.getCandidateSimilarity(context2_emb, candidate_embeddings)
        features.extend(con2_sims)

        # get mention string similarity, todo: no att
        ment_embs = self.getTokenEmbedding(mention_tokens, candidate_embeddings=candidate_embeddings)
        mention_sims = self.getCandidateSimilarity(ment_embs, candidate_embeddings)
        features.extend(mention_sims)

        # neibor mention string similarity
        neigh_ment_sims = DEFAULT_SIM, DEFAULT_SIM, DEFAULT_SIM
        if self._neighbor_window > 0 and num_mentions is not None:
            # (batch * cand_num) * dim
            neigh_ment_embs = self.getNeighborMentionEmbeddings(ment_embs,
                                self._neighbor_window, num_mentions)
            neigh_ment_sims = self.getCandidateSimilarity(neigh_ment_embs, candidate_embeddings)
        features.extend(neigh_ment_sims)

        # neighbor candidates
        # (batch * cand) * (cand_num*window*2+1)
        adj = self.buildGraph(cand_emb1, self._neighbor_window, num_mentions)
        # (batch * cand) * 1 * (2*window*cand_num+1)
        adj = adj.unsqueeze(1)
        # feature vec : (batch * cand) * feature_dim
        h = torch.cat(features, dim=1)
        if self._use_embedding_feature:
            con1_emb_cand1, con1_emb_cand2, con1_emb_cand3 = context1_emb
            h = torch.cat((h, cand_emb1, con1_emb_cand1), dim=1)
            if con2_emb_cand1 is not None:
                h = torch.cat((h, con2_emb_cand1), dim=1)
        if self._gc_ln:
            h = self.ln_inp(h)
        for i in range(self._num_layers):
            layer = getattr(self, 'l{}'.format(i))
            dim = getattr(self, 'd{}'.format(i))
            # (batch_size * cand_num) * (2*window*cand_num+1) * f_dim
            h = self.getExpandFeature(h, self._neighbor_window, num_mentions)
            h = layer(h, adj)
            h = F.relu(h)
            # h: (batch_size * cand_num) * feature_dim
            if length_mask is not None:
                mask = length_mask.view(-1).unsqueeze(1).expand(batch_size*cand_num, dim)
                h = h * mask
        h = F.dropout(h, self._dropout_rate, training=self.training)
        h = self.getExpandFeature(h, self._neighbor_window, num_mentions)
        h = self.gc_classifier(h, adj).squeeze()

        # reshape, batch_size * cand_num
        h = h.view(batch_size, -1)

        output = masked_softmax(h, mask=length_mask)
        return output

    def reset_parameters(self):
        if self.mlp_layer is not None:
            self.mlp_layer.reset_parameters()
        self.classifier.reset_parameters()

# length: batch_size
def sequence_mask(sequence_length, max_length):
    batch_size = sequence_length.size()[0]
    seq_range = torch.arange(0, max_length).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1)
    return seq_range_expand < seq_length_expand


# batch * cand_num
def masked_softmax(logits, mask=None):
    if mask is not None:
        logits = logits * mask
    probs = F.softmax(logits, dim=1)
    return probs