# -*- coding: utf-8 -*-
import numpy as np
import math
# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from ncel.utils.layers import Embed, to_gpu, MLPClassifier, Linear, SubGraphConvolution
from ncel.utils.layers import UniInitializer, LayerNormalization
DEFAULT_SIM = 0.0

def build_model(base_feature_dim, initial_embeddings, FLAGS, logger):
    model_cls = SUBNCEL
    # todo: mlp layer must lager than 1
    mlp_layers_dim = [2000, 1]
    gc_layers = [[1, 1]]
    use_contexts2 = FLAGS.use_lr_context
    use_att = FLAGS.att
    use_embedding_feature = True
    neighbor_ment_window = 3
    neighbor_cand_window = 3
    bias = True
    rho = 0.1
    sim_thred = 0.85
    temperature = 0.2
    return model_cls(
        base_feature_dim,
        initial_embeddings,
        mlp_layers_dim=mlp_layers_dim,
        gc_layers=gc_layers,
        bias=bias,
        ln=FLAGS.mlp_ln,
        dropout=FLAGS.dropout,
        fine_tune_loaded_embeddings=FLAGS.fine_tune_loaded_embeddings,
        use_contexts2=use_contexts2,
        use_att=use_att,
        use_embedding_feature=use_embedding_feature,
        neighbor_ment_window=neighbor_ment_window,
        neighbor_cand_window=neighbor_cand_window,
        rho=rho,
        sim_thred=sim_thred,
        temperature=temperature,
        logger=logger
    )

class SUBNCEL(nn.Module):
    def __init__(self,
                 base_dim,
                 initial_embeddings,
                 mlp_layers_dim=[],
                 gc_layers=[[1,1]],
                 bias=True,
                 ln=False,
                 dropout=0.0,
                 fine_tune_loaded_embeddings=None,
                 use_contexts2=True,
                 use_att=True,
                 use_embedding_feature=True,
                 neighbor_ment_window=3,
                 neighbor_cand_window=3,
                 rho=1.0,
                 sim_thred=0.8,
                 temperature=0.1,
                 logger=None
                 ):
        super(SUBNCEL, self).__init__()

        self._use_contexts2 = use_contexts2
        self._use_att = use_att
        self._dropout_rate = dropout
        self._neighbor_ment_window = neighbor_ment_window
        self._neighbor_cand_window = neighbor_cand_window
        self._ln = ln
        self._gc_ln = True
        self._rho = rho
        self._logger = logger
        self._thred = sim_thred
        self._adj = None
        self._gc_layers = gc_layers
        self._res_num = len(gc_layers)
        self._mlp_classes = mlp_layers_dim[-1]
        self._temperature = temperature
        self.classifier = Linear(initializer=UniInitializer)(1, 1)

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
        margin_col = to_gpu(torch.zeros(1, cand_num))

        right_mask = to_gpu(torch.from_numpy(num_mentions)).float()
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
    def getNeighborMentionEmbeddings(self, ment_emb, neighbor_window, num_mentions):
        batch_size, cand_num = num_mentions.shape
        _, dim = ment_emb.size()
        left_mask, right_mask = self.getNeighborMask(num_mentions, dim)
        margin_col = to_gpu(torch.zeros(cand_num, dim))

        neibor_ment_entity_emb = self.getNeighborMentionEmbeddingsForCandidate(ment_emb,
                                 margin_col, cand_num, neighbor_window, left_mask, right_mask)
        neibor_ment_entity_emb = Variable(neibor_ment_entity_emb, requires_grad=False)
        neibor_ment_sense_emb = neibor_ment_entity_emb
        neibor_ment_mu_emb = neibor_ment_entity_emb
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
        margin_col = to_gpu(torch.zeros(cand_num, dim))
        left_list = []
        # (batch * cand) * dim
        left_list.append(self.leftNeighbor(emb, cand_num, margin_col, left_mask))
        for i in range(window - 1):
            left_list.append(self.leftNeighbor(left_list[i], cand_num, margin_col, left_mask))
        for i in range(window):
            left_list[i] = self.getExpandNeighCandidates(left_list[i], batch_size, cand_num, dim)
        # (batch * cand) * (window*cand) * dim
        left_cands = torch.cat(left_list, dim=1)

        right_list = []
        right_list.append(self.rightNeighbor(emb, cand_num, margin_col, right_mask))
        for i in range(window - 1):
            right_list.append(self.rightNeighbor(right_list[i], cand_num, margin_col, right_mask))
        for i in range(window):
            right_list[i] = self.getExpandNeighCandidates(right_list[i], batch_size, cand_num, dim)
        # (batch * cand) * (window*cand) * dim
        right_cands = torch.cat(right_list, dim=1)

        # (batch * cand) * (cand_num*window*2) * dim
        neigh_cands = torch.cat((left_cands, right_cands), dim=1)
        return neigh_cands

    # cand_emb: (batch * cand) * dim, tensor
    # adj: (batch_size * cand_num) * (2*window*cand_num+1)
    def buildGraph(self, cand_emb, window, num_mentions, thred=0.0):
        batch_size, cand_num = num_mentions.shape
        # (batch * cand) * (cand_num*window*2) * dim
        neigh_cands = self.getNeighCandidates(cand_emb, window, num_mentions)

        # (batch * cand) * (cand_num*window*2) * dim
        cand_emb_expand = cand_emb.unsqueeze(1).expand(batch_size * cand_num,
                                                       2*window*cand_num, self._dim)
        # (batch * cand) * (cand_num*window*2)
        adj = torch.clamp(F.cosine_similarity(cand_emb_expand, neigh_cands, dim=2), thred, 1)
        if thred > 0.0:
            adj[adj<=thred]=0.0
        # add self connection
        margin_col = to_gpu(torch.ones(batch_size*cand_num, 1))
        # size: (batch * cand) * (cand_num*window*2+1)
        adj = torch.cat((adj*self._rho, margin_col), dim=1)
        # normalize
        adj = Variable(F.normalize(adj, p=1, dim=1), requires_grad=False)
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
        candidates = to_gpu(Variable(torch.from_numpy(candidates), volatile=not self.training)).long()
        cand_entity_emb = self.run_embed(candidates, 1)
        cand_sense_emb = None
        cand_mu_emb = None
        if candidates_sense is not None and self._has_sense:
            candidates_sense = to_gpu(Variable(torch.from_numpy(candidates_sense), volatile=not self.training)).long()
            cand_sense_emb = self.run_embed(candidates_sense, 2)
            cand_mu_emb = self.run_embed(candidates_sense, 3)
        return cand_entity_emb, cand_sense_emb, cand_mu_emb

    def getTokenEmbedding(self, tokens, candidate_embeddings=None):
        tokens = to_gpu(Variable(torch.from_numpy(tokens), volatile=not self.training)).long()

        if candidate_embeddings is not None:
            cand_entity_emb, cand_sense_emb, cand_mu_emb = candidate_embeddings
            entity_emb = self.getEmbFeatures(tokens, q_emb=cand_entity_emb)
        else:
            entity_emb = self.getEmbFeatures(tokens)
        sense_emb = entity_emb
        mu_emb = entity_emb

        return entity_emb, sense_emb, mu_emb

    def getCandidateSimilarity(self, embeddings, candidate_embeddings, default_sims=None):
        cand_entity_emb, cand_sense_emb, cand_mu_emb = candidate_embeddings
        entity_emb, sense_emb, mu_emb = embeddings

        if default_sims is None:
            batch_size, _ = entity_emb.size()
            default_sims = to_gpu(Variable(torch.FloatTensor([DEFAULT_SIM]*batch_size).unsqueeze(1),
                                           requires_grad=False))

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
        base_feature = to_gpu(Variable(torch.from_numpy(base_feature[:,:,-1]), requires_grad=False)).float()

        return base_feature.squeeze()

    def reset_parameters(self):
        self.mlp_classifier.reset_parameters()
        for i in range(self._res_num):
            l = getattr(self, 'l{}'.format(i))
            l.reset_parameters()
            # skip connection
            sk_layer = getattr(self, 'sk{}'.format(i))
            if sk_layer is not None:
                sk_layer.reset_parameters()
        if self.classifier is not None:
            self.classifier.reset_parameters()

    # e, num_mentions: batch * cand
    def getGraphSample(self, e, num_mentions, entity_vocab, id2wiki_vocab, only_one=False):
        ent_label_vocab = dict(
            [(entity_vocab[id], id2wiki_vocab[id]) for id in entity_vocab if id in id2wiki_vocab])
        ent_label_vocab[0] = 'PAD'
        ent_label_vocab[1] = 'UNK'

        batch_size, cand_num = e.shape
        # graph, (batch * cand) * (cand_num*window*2+1)
        adj = self._adj.data
        # neighbors, (batch * cand) * (cand_num*window*2)
        e_var = to_gpu(Variable(torch.from_numpy(e).view(-1).unsqueeze(1), requires_grad=False).float())
        neighbors = self.getNeighCandidates(e_var, self._neighbor_cand_window, num_mentions).data.squeeze()
        c_idx = -1
        docs = []
        doc_edges = []
        is_doc_end = False
        for i in range(batch_size):
            for j in range(cand_num):
                c_idx += 1
                if e[i][j] in [0, 1]: continue
                label = ent_label_vocab[e[i][j]]
                # edges
                edges = adj[c_idx]
                nodes = neighbors[c_idx]
                tmp_len = len(edges)-1
                for k in range(tmp_len):
                    if edges[k] > 0 and nodes[k] not in [0, 1]:
                        n_label = ent_label_vocab[nodes[k]]
                        doc_edges.append([label, n_label, edges[k]])
                # doc
                if num_mentions[i][j]==0:
                    is_doc_end = True
            if is_doc_end:
                is_doc_end = False
                doc_line = "Graph: \n" + "\n".join(["{}<-{}->{}".format(edge[0], edge[2],
                                   edge[1]) for edge in doc_edges]) + '\n'
                docs.append(doc_line)
                if only_one: return docs
                del doc_edges[:]
        return docs

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
    probs = F.softmax(logits, dim=1)
    if mask is not None:
        probs = probs * mask
    return probs
