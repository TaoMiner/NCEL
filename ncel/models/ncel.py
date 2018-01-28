# -*- coding: utf-8 -*-
import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from ncel.utils.layers import GraphConvolutionNetwork, Embed, MLP, MLPClassifier, to_gpu


def build_model(initial_embeddings, feature_dim, FLAGS):
    model_cls = NCEL
    num_class_output = 2
    return model_cls(
        initial_embeddings,
        FLAGS.embedding_dim,
        feature_dim,
        FLAGS.mlp_dim,
        FLAGS.gc_dim,
        FLAGS.classifier_dim,
        num_class=num_class_output,
        num_mlp_layers=FLAGS.num_mlp_layers,
        mlp_ln=FLAGS.mlp_ln,
        num_gc_layer=FLAGS.num_gc_layer,
        gc_ln=FLAGS.gc_ln,
        num_cm_layer=FLAGS.num_cm_layer,
        cm_ln=FLAGS.cm_ln,
        dropout=FLAGS.dropout,
        fine_tune_loaded_embeddings=FLAGS.fine_tune_loaded_embeddings
    )


class NCEL(nn.Module):

    def __init__(self,
                 entity_embeddings,
                 embedding_dim,
                 input_dim, # feature_dim
                 mlp_dim,
                 gc_dim,
                 classifier_dim,
                 num_class=2,
                 num_mlp_layers=1,
                 mlp_ln=False,
                 num_gc_layer=2,
                 gc_ln=False,
                 num_cm_layer=1,
                 cm_ln=False,
                 dropout = 0.0,
                 fine_tune_loaded_embeddings=False
                 ):
        super(NCEL, self).__init__()

        entity_vocab_size = entity_embeddings.shape[0]

        self.entity_embed = Embed(
            embedding_dim,
            entity_vocab_size,
            vectors=entity_embeddings,
            fine_tune=fine_tune_loaded_embeddings)


        self.mlp = MLP(input_dim, mlp_dim, num_mlp_layers, mlp_ln,
                       dropout) if num_mlp_layers > 0 else None

        self.gc = GraphConvolutionNetwork(mlp_dim, gc_dim, gc_ln=gc_ln, bias=True,
            num_layers=num_gc_layer, dropout=dropout) if num_gc_layer>0 else None

        self.classifer_mlp = MLPClassifier(gc_dim, classifier_dim, num_class, num_cm_layer,
                                 mlp_ln=cm_ln, classifier_dropout_rate=dropout)

        self.embedding_dim = embedding_dim
        self._num_class = num_class
        self._feature_dim = input_dim

        # For sample printing and logging
        self.mask_memory = None
        self.inverted_vocabulary = None
        self.temperature_to_display = 0.0

    def buildGraph(self, candidates, mask=None):
        batch_size, node_num = candidates.shape
        candidates = to_gpu(Variable(torch.from_numpy(candidates), volatile=not self.training))

        embeds = self.entity_embed(candidates)
        embeds = embeds.view(batch_size, -1, self.embedding_dim)
        if not isinstance(mask, type(None)):
            embed_mask = mask.unsqueeze(2).expand(batch_size, node_num, self.embedding_dim)
            embed_mask = embed_mask.float()
            embeds = embeds * embed_mask
        # batch * node_num * embedding_dim
        # todo: cosine similarity instead of dot product
        adj = torch.bmm(embeds, torch.transpose(embeds, 1, 2))
        adj = adj.clamp(min=0)
        # adj: batch * node * node
        adj = normalize(adj)
        return adj

    # x: batch_size * node_num * feature_dim
    # candidate_ids: batch_size * node_num
    # length: batch_size
    def forward(self, x, candidate_ids, length):
        batch_size, node_num, feature_dim = x.shape
        x = to_gpu(Variable(torch.from_numpy(x), requires_grad=False)).float()

        lengths_var = to_gpu(Variable(torch.from_numpy(length))).long()
        # batch_size * node_num
        length_mask = sequence_mask(lengths_var, node_num)
        adj = self.buildGraph(candidate_ids,
                              mask=length_mask) if not isinstance(self.gc, type(None)) else None
        # adj: batch * node_num * node_num

        h = self.mlp(x, mask=length_mask) if not isinstance(self.mlp, type(None)) else x
        h = self.gc(h, adj, mask=length_mask) if not isinstance(self.gc, type(None)) else h
        # h: batch * node_num * gc_hidden
        batch_size, node_num, gc_hidden = h.size()
        output = self.classifer_mlp(h)
        mask = length_mask.unsqueeze(2).expand(batch_size, node_num, self._num_class)
        mask = mask.float()
        # batch_size * node_num * self._num_class
        output = masked_softmax(output, mask=mask)
        return output

# mx: batch_size * node * node
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = torch.sum(mx, 2)
    batch_size, node_num = rowsum.size()
    r_inv = torch.pow(rowsum, -0.5).view(-1)
    r_inv[r_inv!=r_inv] = 0.
    # r_inv[np.isnan(r_inv)] = 0.
    r_inv = r_inv.view(batch_size, node_num)
    r_mat_inv = torch.stack([torch.diag(d) for d in r_inv], 0)
    # r_mat_inv: batch_size * node * node
    norm_mx = torch.bmm(torch.bmm(r_mat_inv, mx), r_mat_inv)
    return norm_mx

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

def masked_softmax(logits, mask=None):
    probs = F.softmax(logits, dim=2)
    if mask is not None:
        probs = probs * mask
    return probs