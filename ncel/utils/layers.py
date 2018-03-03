# -*- coding: utf-8 -*-
import math

import torch
from torch.autograd import Variable

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.init import kaiming_normal, uniform

import numpy as np
import numba

def the_gpu():
    return the_gpu.gpu

the_gpu.gpu = -1

def to_gpu(var):
    if the_gpu.gpu >= 0:
        return var.cuda(the_gpu.gpu)
    return var

class LayerNormalization(nn.Module):
    # From: https://discuss.pytorch.org/t/lstm-with-layer-normalization/2150

    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a2 = nn.Parameter(torch.ones(1, hidden_size), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)

    def forward(self, z):
        mu = torch.mean(z)
        sigma = torch.std(z)
        ln_out = (z - mu) / (sigma + self.eps)
        ln_out = ln_out * self.a2 + self.b2
        return ln_out

class GraphConvolutionNetwork(Module):
    def __init__(self, input_dim, output_dim, layers_dim=[], gc_ln=False, bias=True, dropout=0.0):
        super(GraphConvolutionNetwork, self).__init__()

        self.dropout_rate = dropout
        self.gc_ln = gc_ln

        if self.gc_ln:
            self.ln_inp = LayerNormalization(input_dim)

        self._layers_dim = layers_dim
        self._layers_dim.append(output_dim)
        self._num_layers = len(self._layers_dim)
        features_dim = input_dim
        for i in range(self._num_layers):
            hidden_dim = layers_dim[i]
            setattr(self, 'l{}'.format(i), GraphConvolution(features_dim, hidden_dim, bias=bias))
            setattr(self, 'd{}'.format(i), hidden_dim)
            features_dim = hidden_dim

    def forward(self, h, adj, mask=None):
        batch_size, node_num, feature_dim = h.size()
        if self.gc_ln:
            h = self.ln_inp(h)
        h = F.dropout(h, self.dropout_rate, training=self.training)
        for i in range(self._num_layers):
            layer = getattr(self, 'l{}'.format(i))
            dim = getattr(self, 'd{}'.format(i))
            h = layer(h, adj)
            h = F.relu(h)
            if mask is not None:
                mask = mask.unsqueeze(1).expand(batch_size, dim)
                h = h * mask
        return h

    def reset_parameters(self):
        for i in range(self.num_layers):
            layer = getattr(self, 'l{}'.format(i))
            layer.reset_parameters()

MAX_DIFF_RES = 128
class ResGraphConvolution(Module):
    """
        n GCN layer with residual unit
        """
    def __init__(self, input_dim, hidden_dim, gc_ln=False, bias=True,
            num_layers=3, dropout=0.0):
        super(ResGraphConvolution, self).__init__()

        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.hidden_dim = hidden_dim
        self.gc_ln = gc_ln

        if self.gc_ln:
            self.ln_inp = LayerNormalization(input_dim)

        features_dim = input_dim
        layer_diff = features_dim - hidden_dim
        is_dim_increase = False
        if layer_diff > 0 and layer_diff > MAX_DIFF_RES:
            layer_diff = MAX_DIFF_RES
        elif layer_diff < 0 and layer_diff < -MAX_DIFF_RES:
            layer_diff = -MAX_DIFF_RES
            is_dim_increase = True
        layer_dim = features_dim - layer_diff

        for i in range(num_layers):
            if (not is_dim_increase and layer_dim < hidden_dim) or \
                    (is_dim_increase and layer_dim > hidden_dim) \
                        or i==num_layers-1: layer_dim=hidden_dim
            setattr(self, 'l{}'.format(i), GraphConvolution(features_dim, layer_dim, bias=bias))
            setattr(self, 'f{}'.format(i), layer_dim)
            if self.gc_ln:
                setattr(self, 'ln{}'.format(i), LayerNormalization(layer_dim))
            features_dim = layer_dim
            layer_dim -= layer_diff

        self.skip_connect_layer = Linear()(input_dim, hidden_dim) if input_dim != hidden_dim else None

    def forward(self, input, adj, mask=None):
        batch_size, node_num, feature_dim = input.size()
        h = self.ln_inp(input) if self.gc_ln else input
        h = F.dropout(h, self.dropout_rate, training=self.training)
        for i in range(self.num_layers):
            layer = getattr(self, 'l{}'.format(i))
            h = layer(h, adj)
            h = F.relu(h)
            if not isinstance(mask, type(None)):
                f = getattr(self, 'f{}'.format(i))
                gc_mask = mask.unsqueeze(2).expand(batch_size, node_num, f)
                gc_mask = gc_mask.float()
                h = h * gc_mask
            if self.gc_ln:
                ln = getattr(self, 'ln{}'.format(i))
                h = ln(h)
            h = F.dropout(h, self.dropout_rate, training=self.training)
        if self.skip_connect_layer is not None:
            h = h + self.skip_connect_layer(input)
        else:
            h = h + input
        if not isinstance(mask, type(None)):
            gc_mask = mask.unsqueeze(2).expand(batch_size, node_num, self.hidden_dim)
            gc_mask = gc_mask.float()
            h = h * gc_mask
        return h

    def reset_parameters(self):
        for i in range(self.num_layers):
            layer = getattr(self, 'l{}'.format(i))
            layer.reset_parameters()
        if self.skip_connect_layer is not None:
            self.skip_connect_layer.reset_parameters()

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # input: batch_size * node_num * in_features
    # adj : batch_size * node_num * node_num
    def forward(self, input, adj):
        support = input.matmul(self.weight)
        output = torch.bmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class SubGraphConvolution(Module):
    """
    partial GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, layer_dims=[100], bias=True,
                 initializer=UniInitializer, bias_initializer=UniInitializer):
        super(SubGraphConvolution, self).__init__()
        self._layer_dims = layer_dims
        self._initializer = initializer
        self._bias_initializer = bias_initializer
        self._in_features = in_features

        features_dim = self._in_features
        for i, dim in enumerate(self._layer_dims):
            hidden_dim = dim
            setattr(self, 'w{}'.format(i), Parameter(torch.Tensor(features_dim, hidden_dim)))
            setattr(self, 'b{}'.format(i), Parameter(torch.Tensor(hidden_dim)) if bias else None)
            features_dim = hidden_dim
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self._layer_dims)):
            lw = getattr(self, 'w{}'.format(i))
            lb = getattr(self, 'b{}'.format(i))
            self._initializer(lw)
            self._bias_initializer(lb)

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

    def getExpandNeighCandidates(self, neigh_emb, batch_size, cand_num, dim):
        tmp_neigh_cand = neigh_emb.view(batch_size, cand_num, dim)
        tmp_neigh_cand = tmp_neigh_cand.unsqueeze(1).expand(batch_size, cand_num,
                                                            cand_num, dim)
        # (batch * cand) * cand * dim
        tmp_neigh_cand = tmp_neigh_cand.contiguous().view(batch_size * cand_num, cand_num, dim)
        return tmp_neigh_cand

    def getNeighCandidates(self, emb, window, graph_boundry):
        batch_size, cand_num = graph_boundry.shape
        _, dim = emb.size()
        left_mask, right_mask = self.getNeighborMask(graph_boundry, dim)
        margin_col = to_gpu(Variable(torch.zeros(cand_num, dim), requires_grad=False))
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

    # f: (batch * group) * dim
    # gf_neighbors: (batch * group) * (2*group*window+1) * dim
    def getExpandFeature(self, f, window, graph_boundry):
        _, f_dim = f.size()
        # (batch * group) * (2*group*window) * dim
        gf_neighbors = self.getNeighCandidates(f, window, graph_boundry)
        # (batch * group) * (2*group*window+1) * dim
        gf_neighbors = torch.cat((gf_neighbors, f.unsqueeze(1)), dim=1)
        return gf_neighbors

    # x: batch * group * in_features
    # adj : batch * group * (2*group*window+1), self connection
    # mask : batch * group
    # graph_boundry : numpy, batch * group, 0 indicates the last group in the graph, otherwise 1
    def forward(self, x, adj, graph_boundry, mask=None):
        batch_size, group_size, neighbor_size = adj.size()
        # todo: more flexible for adj neighbors
        adj = adj.view(batch_size*group_size, neighbor_size).unsqueeze(1)
        window = (neighbor_size-1)/group_size/2

        h = x.view(batch_size*group_size, -1)

        for i, dim in enumerate(self._layer_dims):
            w = getattr(self, 'w{}'.format(i))
            b = getattr(self, 'b{}'.format(i))
            h = h.matmul(w)
            # (batch * group) * (2*group*window+1) * dim
            h = self.getExpandFeature(h, window, graph_boundry)
            h = torch.bmm(adj, h).squeeze(1)
            if b is not None: h = h + b
            # h: batch * 1
            if mask is not None:
                mask = mask.view(-1).unsqueeze(1).expand(batch_size*group_size, dim)
                h = h * mask
            h = F.relu(h)
        return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self._in_features) + ' -> '.join([str(d) for d in self._layer_dims]) + ')'

class MLPClassifier(nn.Module):
    def __init__(self,
                 input_dim,
                 num_class,
                 layers_dim=[],
                 mlp_ln=False,
                 dropout=0.0
                 ):
        super(MLPClassifier, self).__init__()

        self.drop_out_rate = dropout
        self.mlp_ln = mlp_ln
        self.num_class = num_class

        if self.mlp_ln:
            self.ln_inp = LayerNormalization(input_dim)

        self._mlp_layer_num = len(layers_dim)

        if self._mlp_layer_num > 0:
            self.mlp_layer = MLP(input_dim, layers_dim[-1], layers_dim=layers_dim[:-1])
            self.classifier = Linear(initializer=UniInitializer)(layers_dim[-1], num_class)
        else:
            self.mlp_layer = None
            self.classifier = Linear(initializer=UniInitializer)(input_dim, num_class)

    def forward(self, h, length=None):
        batch_size, _ = h.size()
        if self.mlp_ln:
            h = self.ln_inp(h)
        if self.mlp_layer is not None:
            h = self.mlp_layer(h, length=length)
        h = F.dropout(h, self.drop_out_rate, training=self.training)
        h = self.classifier(h)
        if length is not None:
            mask = length.unsqueeze(1).expand(batch_size, self.num_class)
            h = h * mask
        if self.num_class == 1:
            h = h.squeeze()
        else:
            h = F.softmax(h)
        return h

    def reset_parameters(self):
        if self.mlp_layer is not None:
            self.mlp_layer.reset_parameters()
        self.classifier.reset_parameters()

class MLP(nn.Module):
    def __init__(
            self,
            mlp_input_dim,
            output_dim,
            layers_dim=[]):
        super(MLP, self).__init__()

        layers_dim.append(output_dim)
        self.num_mlp_layers = len(layers_dim)

        features_dim = mlp_input_dim
        for i in range(self.num_mlp_layers):
            hidden_dim = layers_dim[i]
            setattr(self, 'l{}'.format(i), Linear()(features_dim, hidden_dim))
            setattr(self, 'd{}'.format(i), hidden_dim)
            features_dim = hidden_dim

    def forward(self, h, length=None):
        batch_size, _ = h.size()
        for i in range(self.num_mlp_layers):
            layer = getattr(self, 'l{}'.format(i))
            dim = getattr(self, 'd{}'.format(i))
            h = layer(h)
            if length is not None:
                mask = length.unsqueeze(1).expand(batch_size, dim)
                h = h * mask
            h = F.relu(h)
        return h

    def reset_parameters(self):
        for i in range(self.num_mlp_layers):
            layer = getattr(self, 'l{}'.format(i))
            layer.reset_parameters()

def ZeroInitializer(param):
    shape = param.size()
    init = np.zeros(shape).astype(np.float32)
    param.data.set_(torch.from_numpy(init))

def UniInitializer(param):
    uniform(param, -0.005, 0.005)

def Linear(initializer=kaiming_normal,
           bias_initializer=ZeroInitializer):
    class CustomLinear(nn.Linear):
        def reset_parameters(self):
            initializer(self.weight)
            if self.bias is not None:
                bias_initializer(self.bias)
    return CustomLinear

class Embed(nn.Module):
    def __init__(self, size, vocab_size, vectors, fine_tune=False):
        super(Embed, self).__init__()
        if fine_tune:
            self.embed = nn.Embedding(vocab_size, size, sparse=False)
            self.embed.weight.data.copy_(torch.from_numpy(vectors))
        else:
            self.vectors = vectors
            self.embed = None

    # tokens: batch_size * token_num
    # embeds: (batch_size * token_num) * embedding_dim
    def forward(self, tokens):
        if self.embed is not None:
            embeds = self.embed(tokens.contiguous().view(-1).long())
        else:
            embeds = self.vectors.take(
                tokens.data.cpu().numpy().ravel(), axis=0)
            embeds = to_gpu(Variable(
                    torch.from_numpy(embeds),
                    volatile=tokens.volatile))
        return embeds

@numba.autojit
def cosSim(v1, v2):
    res = 0
    len_v1 = math.sqrt(np.dot(v1, v1))
    len_v2 = math.sqrt(np.dot(v2, v2))
    if len_v1 > 0.000001 and len_v2 > 0.000001:
        res = np.dot(v1, v2) / len_v1 / len_v2
        # res = (res + 1) / 2
    if math.isnan(res) or math.isinf(res) or res < -1:
        res = -1
    elif res > 1 : res = 1
    return res

# input all candidates in one document, return one graph much like self attention
# todo: no mention group split
def buildGraph(ids, embeddings, thred=0):
    node_num = ids.shape[0]
    adj = np.zeros((node_num, node_num))
    # node * dim
    embeds = embeddings.take(np.array([cid[0] for cid in ids]).ravel(), axis=0)

    for i, ei in enumerate(embeds):
        for j, ej in enumerate(embeds):
            if i == j: adj[i][j] = 1.0
            elif i > j or ids[i][1] == ids[j][1]: continue
            else:
                tmp_sim = cosSim(ei, ej)
                if tmp_sim < thred : tmp_sim = 0
                adj[i][j] = adj[j][i] = tmp_sim

    adj = normalize(adj)
    return adj

# input all candidates in one document, return one graph much like self attention
@numba.autojit
def buildFullGraph(ids, embeddings, thred=0):
    node_num = ids.shape[0]
    # node * dim
    embeds = embeddings.take(np.array([cid[0] for cid in ids]).ravel(), axis=0)

    adj = np.dot(embeds, embeds.transpose())
    adj[ adj<thred ] = 0
    adj = normalize(adj)
    return adj

@numba.autojit
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isnan(r_inv)] = 0.
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = np.dot(mx, r_mat_inv)
    norm_mx = np.dot(mx.transpose(), r_mat_inv)
    return norm_mx