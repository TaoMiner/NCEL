# -*- coding: utf-8 -*-
import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from ncel.utils.layers import Embed, to_gpu, MLPClassifier


def build_model(base_feature_dim, initial_embeddings, FLAGS):
    model_cls = MLPC
    layers_dim = [256]
    use_contexts2 = FLAGS.use_lr_context
    use_entity = False
    use_att = FLAGS.att
    return model_cls(
        base_feature_dim,
        initial_embeddings,
        layers_dim=layers_dim,
        mlp_ln=FLAGS.mlp_ln,
        dropout=FLAGS.dropout,
        fine_tune_loaded_embeddings=FLAGS.fine_tune_loaded_embeddings,
        use_contexts2=use_contexts2,
        use_entity=use_entity,
        use_att=use_att
    )


class MLPC(nn.Module):

    def __init__(self,
                 base_dim,
                 initial_embeddings,
                 layers_dim=[],
                 mlp_ln=False,
                 dropout = 0.0,
                 fine_tune_loaded_embeddings=None,
                 use_contexts2=True,
                 use_entity=False,
                 use_att=True
                 ):
        super(MLPC, self).__init__()

        self._use_contexts2 = use_contexts2
        self._use_entity = use_entity
        self._use_att = use_att

        word_embeddings, entity_embeddings, sense_embeddings, mu_embeddings = initial_embeddings
        word_vocab_size, word_embedding_dim = word_embeddings.shape
        entity_vocab_size, entity_embedding_dim = entity_embeddings.shape
        sense_vocab_size, sense_embedding_dim = sense_embeddings.shape

        self.word_embed = Embed(word_embedding_dim, word_vocab_size,
                            vectors=word_embeddings, fine_tune=fine_tune_loaded_embeddings)

        self.entity_embed = None
        if use_entity:
            self.entity_embed = Embed(entity_embedding_dim, entity_vocab_size,
                            vectors=entity_embeddings, fine_tune=fine_tune_loaded_embeddings)

        self.sense_embed = Embed(sense_embedding_dim, sense_vocab_size,
                                  vectors=sense_embeddings, fine_tune=fine_tune_loaded_embeddings)
        self.mu_embed = Embed(sense_embedding_dim, sense_vocab_size,
                                 vectors=mu_embeddings, fine_tune=fine_tune_loaded_embeddings)

        self.embeds = [self.word_embed, self.sense_embed, self.mu_embed, self.entity_embed]

        # base_dim + sense_dim + word_dim + 2 + 1(if has entity) + (2+word_dim)(if has contexts) + 1(if has context2 and has entity)
        self._feature_dim = base_dim + sense_embedding_dim + word_embedding_dim + 2
        if self._use_entity:
            self._feature_dim += 1
        if self._use_contexts2:
            self._feature_dim += 2 + word_embedding_dim
            if self._use_entity:
                self._feature_dim += 1

        self.mlp_classifier = MLPClassifier(self._feature_dim, 1, layers_dim=layers_dim,
                                            mlp_ln=mlp_ln, dropout=dropout)

    # types: index of [word,sense,mu,entity]
    def run_embed(self, x, type):
        embeds = self.embeds[type](x)
        embeds = F.dropout(embeds, self.embedding_dropout_rate, training=self.training)
        return embeds

    def getEmbFeatures(self, sents, q_emb=None):
        batch_size, cand_num, seq_length = sents.size()

        sents_emb = self.run_embed(sents, 0)
        sents_emb = sents_emb.view(batch_size * cand_num, seq_length, -1)

        if self._use_att and q_emb is not None:
            att = torch.bmm(q_emb.unsqueeze(1), sents_emb.transpose(1, 2))
            f_emb = torch.bmm(att, sents_emb).squeeze()
        else:
            f_emb = torch.mean(sents_emb, dim=1)

        return f_emb

    # contexts1 : batch * candidate * tokens
    # contexts2 : batch * candidate * tokens
    # base_feature : batch * candidate * features, numpy
    # candidates : batch * candidate
    # candidates_entity: batch * candidate
    # length: batch
    def forward(self, contexts1, base_feature, candidates,
                contexts2=None, candidates_entity=None, length=None):
        batch_size, cand_num, _ = base_feature.shape
        # to gpu
        base_feature = to_gpu(Variable(torch.from_numpy(base_feature)))
        contexts1 = to_gpu(Variable(torch.from_numpy(contexts1))).long()
        candidates = to_gpu(Variable(torch.from_numpy(candidates))).long()

        has_context2 = False
        if contexts2 is not None and self._use_contexts2:
            contexts2 = to_gpu(Variable(torch.from_numpy(contexts2))).long()
            has_context2 = True

        has_entity = False
        if candidates_entity is not None and self._use_entity:
            candidates_entity = to_gpu(Variable(torch.from_numpy(candidates_entity))).long()
            has_entity = True

        # get emb, (batch * cand) * dim
        cand_emb = self.run_embed(candidates, 1)
        cand_mu_emb = self.run_embed(candidates, 2)

        f1_sense_emb = self.getEmbFeatures(contexts1, q_emb=cand_emb)
        f1_mu_emb = self.getEmbFeatures(contexts1, q_emb=cand_mu_emb)
        if has_entity:
            cand_entity_emb = self.run_embed(candidates_entity, 3)
            f1_entity_emb = self.getEmbFeatures(contexts1, q_emb=cand_entity_emb)

        if has_context2:
            f2_sense_emb = self.getEmbFeatures(contexts2, q_emb=cand_emb)
            f2_mu_emb = self.getEmbFeatures(contexts2, q_emb=cand_mu_emb)
            if has_entity:
                f2_entity_emb = self.getEmbFeatures(contexts2, q_emb=cand_entity_emb)

        # get contextual similarity, (batch * cand) * contextual_sim
        cand_emb_expand = cand_emb.unsqueeze(1)
        cand_mu_emb_expand = cand_mu_emb.unsqueeze(1)
        # sense : context1
        sim1 = torch.bmm(cand_emb_expand, f1_sense_emb.unsqueeze(2))
        # mu : context1
        sim2 = torch.bmm(cand_mu_emb_expand, f1_mu_emb.unsqueeze(2))
        # entity: context1
        if has_entity:
            cand_entity_emb_expand = cand_entity_emb.unsqueeze(1)
            sim3 = torch.bmm(cand_entity_emb_expand, f1_entity_emb.unsqueeze(2))

        # sense : context2
        if has_context2:
            sim4 = torch.bmm(cand_emb_expand, f2_sense_emb.unsqueeze(2))
            # mu : context2
            sim5 = torch.bmm(cand_mu_emb_expand, f2_mu_emb.unsqueeze(2))
            # entity: context2
            if has_entity:
                sim6 = torch.bmm(cand_entity_emb_expand, f2_entity_emb.unsqueeze(2))

        # feature vec : batch * cand * feature_dim
        # feature dim: base_dim + sense_dim + word_dim + 2 + 1(if has entity) +
        # (2+word_dim)(if has contexts) + 1(if has context2 and has entity)
        h = torch.cat((base_feature, cand_emb, f1_sense_emb, sim1, sim2), dim=1)
        if has_entity:
            h = torch.cat((h, sim3), dim=1)
        if has_context2:
            h = torch.cat((h, sim4, sim5, f2_sense_emb), dim=1)
            if has_entity:
                h = torch.cat((h, sim6), dim=1)

        if self.mlp_ln:
            h = self.ln_inp(h)
        if self.mlp_layer is not None:
            h = self.mlp_layer(h)
        h = F.dropout(h, self.drop_out_rate, training=self.training)
        h = self.classifier(h)
        # reshape, batch_size * cand_num
        h = h.view(batch_size, -1)

        if length is not None:
            lengths_var = to_gpu(Variable(torch.from_numpy(length), requires_grad=False)).long()
            # batch_size * cand_num
            length_mask = sequence_mask(lengths_var, cand_num).float()

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
    probs = F.softmax(logits, dim=1)
    if mask is not None:
        probs = probs * mask
    return probs