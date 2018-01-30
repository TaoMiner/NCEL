# -*- coding: utf-8 -*-
import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from ncel.utils.layers import GraphConvolutionNetwork, MLP, to_gpu
from ncel.utils.layers import Linear, UniInitializer

def build_model(feature_dim, FLAGS):
    model_cls = NCEL
    return model_cls(
        feature_dim,
        FLAGS.mlp_dim,
        FLAGS.gc_dim,
        num_mlp_layers=FLAGS.num_mlp_layers,
        mlp_ln=FLAGS.mlp_ln,
        num_gc_layer=FLAGS.num_gc_layer,
        gc_ln=FLAGS.gc_ln,
        dropout=FLAGS.dropout,
        res_gc_layer_num=FLAGS.res_gc_layer_num
    )


class NCEL(nn.Module):

    def __init__(self,
                 input_dim, # feature_dim
                 mlp_dim,
                 gc_dim,
                 num_mlp_layers=1,
                 mlp_ln=False,
                 num_gc_layer=3,
                 gc_ln=False,
                 dropout = 0.0,
                 res_gc_layer_num=3
                 ):
        super(NCEL, self).__init__()

        self.bias = True
        hidden_dim = input_dim
        self.mlp = None
        if num_mlp_layers > 0:
            self.mlp = MLP(input_dim, mlp_dim, num_mlp_layers, mlp_ln, dropout)
            hidden_dim = mlp_dim

        self.gc_layer = None
        if num_gc_layer > 0:
            self.gc_layer = GraphConvolutionNetwork(hidden_dim, gc_dim, gc_ln=gc_ln, bias=self.bias,
                num_layers=num_gc_layer, dropout=dropout, res_gc_layer_num=res_gc_layer_num)
            hidden_dim = gc_dim

        self.classifer = Linear(initializer=UniInitializer)(hidden_dim, 1)

    # x: batch_size * node_num * feature_dim
    # adj: batch_size * node_num * node_num
    # length: batch_size
    def forward(self, x, length=None, adj=None):
        batch_size, node_num, feature_dim = x.shape
        h = to_gpu(Variable(torch.from_numpy(x), requires_grad=False)).float()
        length_mask = None
        if length is not None:
            lengths_var = to_gpu(Variable(torch.from_numpy(length), requires_grad=False)).long()
            # batch_size * node_num
            length_mask = sequence_mask(lengths_var, node_num)
        # adj: batch * node_num * node_num
        h = self.mlp(h, mask=length_mask) if not isinstance(self.mlp, type(None)) else h

        if not isinstance(self.gc_layer, type(None)) and adj is not None:
            adj = to_gpu(Variable(torch.from_numpy(adj), requires_grad=False)).float()
            h = self.gc_layer(h, adj, mask=length_mask)
        # h: batch * node_num * hidden
        batch_size, node_num, _ = h.size()
        output = self.classifer(h)
        # batch_size * node_num * self._num_class
        output = masked_softmax2d(output, mask=length_mask)
        return output

    def reset_parameters(self):
        if self.mlp is not None:
            self.mlp.reset_parameters()
        if self.gc_layer is not None:
            self.gc_layer.reset_parameters()
        self.classifer.reset_parameters()


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

def masked_softmax2d(logits, mask=None):
    probs = F.softmax(logits.squeeze(), dim=1)
    if mask is not None:
        mask = mask.float()
        probs = probs * mask
    return probs

def masked_softmax(logits, mask=None):
    probs = F.softmax(logits, dim=2)
    if mask is not None:
        probs = probs * mask
    return probs