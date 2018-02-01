# -*- coding: utf-8 -*-
import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from ncel.utils.layers import GraphConvolutionNetwork, Linear, to_gpu, UniInitializer, LayerNormalization


def build_model(feature_dim, FLAGS):
    model_cls = NCEL
    num_class_output = 2
    return model_cls(
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
        res_gc_layer_num=FLAGS.res_gc_layer_num
    )


class NCEL(nn.Module):

    def __init__(self,
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
                 class_ln=False,
                 dropout = 0.0,
                 res_gc_layer_num=0
                 ):
        super(NCEL, self).__init__()

        self.drop_out_rate = dropout
        self.class_ln = class_ln

        self.gc_layer = GraphConvolutionNetwork(input_dim, gc_dim, gc_ln=gc_ln, bias=True,
            num_layers=num_gc_layer, dropout=dropout, res_gc_layer_num=res_gc_layer_num)

        if self.class_ln:
            self.ln_inp = LayerNormalization(input_dim)

        self.classifer = Linear(initializer=UniInitializer)(gc_dim, 2)

    # x: batch_size * node_num * feature_dim
    # adj: batch_size * node_num * node_num
    # length: batch_size
    def forward(self, x, adj, length=None):
        batch_size, node_num, feature_dim = x.shape
        h = to_gpu(Variable(torch.from_numpy(x), requires_grad=False)).float()

        length_mask = None
        if length is not None:
            lengths_var = to_gpu(Variable(torch.from_numpy(length), requires_grad=False)).long()
            # batch_size * node_num
            length_mask = sequence_mask(lengths_var, node_num)

            class_mask = length_mask.unsqueeze(2).expand(batch_size, node_num, self._num_class)
            class_mask = class_mask.float()
        # adj: batch * node_num * node_num

        adj = to_gpu(Variable(torch.from_numpy(adj), requires_grad=False)).float()
        h = self.gc_layer(h, adj, mask=length_mask)
        # h: batch * node_num * hidden
        if self.class_ln:
            h = self.ln_inp(h)
        h = F.dropout(h, self.drop_out_rate, training=self.training)
        output = self.classifer(h)
        # batch_size * node_num * self._num_class
        output = masked_softmax(output, mask=class_mask)
        return output

    def reset_parameters(self):
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

def masked_softmax(logits, mask=None):
    probs = F.softmax(logits, dim=2)
    if mask is not None:
        probs = probs * mask
    return probs