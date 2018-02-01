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

MAX_DIFF = 128
class GraphConvolutionNetwork(Module):
    def __init__(self, input_dim, hidden_dim, gc_ln=False, bias=True,
            num_layers=1, dropout=0.0, res_gc_layer_num=0):
        super(GraphConvolutionNetwork, self).__init__()

        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.hidden_dim = hidden_dim
        self.gc_ln = gc_ln

        if self.gc_ln:
            self.ln_inp = LayerNormalization(input_dim)

        features_dim = input_dim
        layer_diff = features_dim - hidden_dim
        is_dim_increase = False
        if layer_diff > 0 and layer_diff > MAX_DIFF :
            layer_diff = MAX_DIFF
        elif layer_diff < 0 and layer_diff < -MAX_DIFF :
            layer_diff = -MAX_DIFF
            is_dim_increase = True
        layer_dim = features_dim - layer_diff

        for i in range(num_layers):
            if (not is_dim_increase and layer_dim < hidden_dim) or \
                    (is_dim_increase and layer_dim > hidden_dim) \
                        or i==num_layers-1: layer_dim=hidden_dim
            setattr(self, 'l{}'.format(i), ResGraphConvolution(
                                            features_dim, layer_dim, gc_ln=gc_ln, bias=bias,
                                            num_layers=res_gc_layer_num, dropout=dropout))
            setattr(self, 'f{}'.format(i), layer_dim)
            if self.gc_ln:
                setattr(self, 'ln{}'.format(i), LayerNormalization(layer_dim))
            features_dim = layer_dim
            layer_dim -= layer_diff

    def forward(self, h, adj, mask=None):
        batch_size, node_num, feature_dim = h.size()
        if self.gc_ln:
            h = self.ln_inp(h)
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
    def forward(self, input, adj, mask=None):
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

class MLPClassifier(nn.Module):
    def __init__(
            self,
            mlp_input_dim,
            mlp_dim,
            num_classes,
            num_mlp_layers,
            mlp_ln=False,
            classifier_dropout_rate=0.0):
        super(MLPClassifier, self).__init__()

        self.num_mlp_layers = num_mlp_layers
        self.mlp_ln = mlp_ln
        self.classifier_dropout_rate = classifier_dropout_rate
        self.hidden_dim = mlp_dim
        features_dim = mlp_input_dim

        if mlp_ln:
            self.ln_inp = LayerNormalization(mlp_input_dim)
        if num_mlp_layers > 0:
            layer_diff = int((mlp_input_dim - mlp_dim) / num_mlp_layers)
            layer_dim = features_dim - layer_diff

            for i in range(num_mlp_layers):
                if i == num_mlp_layers - 1: layer_dim = mlp_dim
                setattr(self, 'l{}'.format(i), Linear()(features_dim, layer_dim))
                setattr(self, 'f{}'.format(i), layer_dim)
                if mlp_ln:
                    setattr(self, 'ln{}'.format(i), LayerNormalization(layer_dim))
                features_dim = layer_dim
                layer_dim -= layer_diff

        setattr(
            self,
            'l{}'.format(num_mlp_layers),
            Linear(initializer=UniInitializer)(
                features_dim,
                num_classes))

    def forward(self, h, mask=None):
        batch_size, node_num, feature_dim = h.size()
        if self.mlp_ln:
            h = self.ln_inp(h)
        h = F.dropout(h, self.classifier_dropout_rate, training=self.training)
        for i in range(self.num_mlp_layers):
            layer = getattr(self, 'l{}'.format(i))
            h = layer(h)
            h = F.relu(h)
            if not isinstance(mask, type(None)):
                f = getattr(self, 'f{}'.format(i))
                mlp_mask = mask.unsqueeze(2).expand(batch_size, node_num, f)
                mlp_mask = mlp_mask.float()
                h = h * mlp_mask
            if self.mlp_ln:
                ln = getattr(self, 'ln{}'.format(i))
                h = ln(h)
            h = F.dropout(
                h,
                self.classifier_dropout_rate,
                training=self.training)
        layer = getattr(self, 'l{}'.format(self.num_mlp_layers))
        y = layer(h)
        return y

    def reset_parameters(self):
        for i in range(self.num_mlp_layers):
            layer = getattr(self, 'l{}'.format(i))
            layer.reset_parameters()
        layer = getattr(self, 'l{}'.format(self.num_mlp_layers))
        layer.reset_parameters()

class MLP(nn.Module):
    def __init__(
            self,
            mlp_input_dim,
            mlp_dim,
            num_mlp_layers,
            mlp_ln=False,
            dropout_rate=0.0):
        super(MLP, self).__init__()

        self.num_mlp_layers = num_mlp_layers
        self.mlp_ln = mlp_ln
        self.dropout_rate = dropout_rate
        self.hidden_dim = mlp_dim
        if mlp_ln:
            self.ln_inp = LayerNormalization(mlp_input_dim)

        features_dim = mlp_input_dim
        if num_mlp_layers > 0:
            layer_diff = int((mlp_input_dim - mlp_dim) / num_mlp_layers)
            layer_dim = features_dim - layer_diff

            for i in range(num_mlp_layers):
                if i == num_mlp_layers - 1: layer_dim = mlp_dim
                setattr(self, 'l{}'.format(i), Linear()(features_dim, layer_dim))
                setattr(self, 'f{}'.format(i), layer_dim)
                if mlp_ln:
                    setattr(self, 'ln{}'.format(i), LayerNormalization(layer_dim))
                features_dim = layer_dim
                layer_dim -= layer_diff

    def forward(self, h, mask=None):
        if self.num_mlp_layers>0:
            batch_size, node_num, feature_dim = h.size()
            if self.mlp_ln:
                h = self.ln_inp(h)
            h = F.dropout(h, self.dropout_rate, training=self.training)
            for i in range(self.num_mlp_layers):
                layer = getattr(self, 'l{}'.format(i))
                h = layer(h)
                h = F.relu(h)
                if not isinstance(mask, type(None)):
                    f = getattr(self, 'f{}'.format(i))
                    mlp_mask = mask.unsqueeze(2).expand(batch_size, node_num, f)
                    mlp_mask = mlp_mask.float()
                    h = h * mlp_mask
                if self.mlp_ln:
                    ln = getattr(self, 'ln{}'.format(i))
                    h = ln(h)
                h = F.dropout(
                    h,
                    self.dropout_rate,
                    training=self.training)
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
            self.embed = nn.Embedding(vocab_size, size, sparse=True)
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