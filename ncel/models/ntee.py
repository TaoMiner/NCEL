# -*- coding: utf-8 -*-
import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from ncel.utils.layers import Linear, to_gpu, UniInitializer, LayerNormalization, MLPClassifier


def build_model(feature_dim, FLAGS):
    model_cls = NTEE
    return model_cls(
        feature_dim,
        FLAGS.mlp_dim,
        num_classes=FLAGS.num_mlp_layer,
        mlp_ln=FLAGS.mlp_ln,
        dropout=FLAGS.dropout
    )


class NTEE(nn.Module):

    def __init__(self,
                 input_dim, # feature_dim
                 mlp_dim,
                 num_classes=1,
                 num_mlp_layer=1,
                 mlp_ln=False,
                 dropout = 0.0
                 ):
        super(NTEE, self).__init__()

        self.drop_out_rate = dropout
        self.mlp_ln = mlp_ln
        self.num_classes = num_classes

        if self.mlp_ln:
            self.ln_inp = LayerNormalization(input_dim)

        self.mlp_layer = MLPClassifier(input_dim, mlp_dim, num_classes,
            num_mlp_layer, mlp_ln=mlp_ln, classifier_dropout_rate=dropout)

    # x: batch_size * node_num * feature_dim
    # length: batch_size
    def forward(self, x, length=None):
        batch_size, node_num, feature_dim = x.shape
        h = to_gpu(Variable(torch.from_numpy(x), requires_grad=False)).float()
        if self.mlp_ln:
            h = self.ln_inp(h)
        length_mask = None
        if length is not None:
            lengths_var = to_gpu(Variable(torch.from_numpy(length), requires_grad=False)).long()
            # batch_size * node_num
            length_mask = sequence_mask(lengths_var, node_num).float()
        h = F.dropout(h, self.drop_out_rate, training=self.training)
        h = self.mlp_layer(h, mask=length_mask)
        # h: batch * node_num * num_class

        # batch_size * node_num
        output = masked_softmax(h, mask=length_mask)
        return output

    def reset_parameters(self):
        self.mlp_layer.reset_parameters()


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
    probs = F.softmax(logits, dim=1)
    if mask is not None:
        probs = probs * mask
    return probs