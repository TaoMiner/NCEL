# -*- coding: utf-8 -*-
import math

import torch

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

class GraphConvolutionNetwork(Module):
    def __init__(self, input_dim, hidden_dim, gc_ln=False, bias=True,
            num_layers=1, dropout=0.0):
        super(GraphConvolutionNetwork, self).__init__()

        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.hidden_dim = hidden_dim

        if gc_ln:
            self.ln_inp = LayerNormalization(input_dim)

        features_dim = input_dim

        for i in range(num_layers):
            setattr(self, 'l{}'.format(i), GraphConvolution(features_dim, hidden_dim, bias=bias))
            if gc_ln:
                setattr(self, 'ln{}'.format(i), LayerNormalization(hidden_dim))
            features_dim = hidden_dim

    def forward(self, input, adj, mask=None):
        batch_size, node_num, feature_dim = input.size()
        if self.gc_ln:
            h = self.ln_inp(input)
        h = F.dropout(input, self.dropout_rate, training=self.training)
        for i in range(self.num_layers):
            layer = getattr(self, 'l{}'.format(i))
            h = layer(input, adj)
            h = F.relu(h)
            if not isinstance(mask, type(None)):
                h = h * mask.unsqueeze(2).expand(batch_size, node_num, self.hidden_dim)
            if self.gc_ln:
                ln = getattr(self, 'ln{}'.format(i))
                h = ln(h)
            h = F.dropout(h, self.dropout_rate, training=self.training)
        return h

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
        support = input.matmul(self.weight.t())
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

        for i in range(num_mlp_layers):
            setattr(self, 'l{}'.format(i), Linear()(features_dim, mlp_dim))
            if mlp_ln:
                setattr(self, 'ln{}'.format(i), LayerNormalization(mlp_dim))
            features_dim = mlp_dim
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
                h = h * mask.unsqueeze(2).expand(batch_size, node_num, self.hidden_dim)
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
        features_dim = mlp_input_dim,
        self.hidden_dim = mlp_dim
        if mlp_ln:
            self.ln_inp = LayerNormalization(mlp_input_dim)

        for i in range(num_mlp_layers):
            setattr(self, 'l{}'.format(i), Linear()(features_dim, mlp_dim))
            if mlp_ln:
                setattr(self, 'ln{}'.format(i), LayerNormalization(mlp_dim))
            features_dim = mlp_dim

    def forward(self, h, mask=None):
        batch_size, node_num, feature_dim = h.size()
        if self.mlp_ln:
            h = self.ln_inp(h)
        h = F.dropout(h, self.dropout_rate, training=self.training)
        for i in range(self.num_mlp_layers):
            layer = getattr(self, 'l{}'.format(i))
            h = layer(h)
            h = F.relu(h)
            if not isinstance(mask, type(None)):
                h = h * mask.unsqueeze(2).expand(batch_size, node_num, self.hidden_dim)
            if self.mlp_ln:
                ln = getattr(self, 'ln{}'.format(i))
                h = ln(h)
            h = F.dropout(
                h,
                self.dropout_rate,
                training=self.training)
        return h

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
        return embeds