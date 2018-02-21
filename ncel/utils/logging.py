# -*- coding: utf-8 -*-
"""
logging.py

Log format convenience methods for training spinn.

"""

from ncel.utils.misc import time_per_token
import numpy as np


class InspectModel(object):
    '''Examines what kind of SPINN model we are dealing with.'''

    def __init__(self, model):
        self.has_spinn = hasattr(model, 'spinn')
        self.has_transition_loss = hasattr(
            model, 'transition_loss') and model.transition_loss is not None
        self.has_invalid = self.has_spinn and hasattr(model.spinn, 'invalid')
        self.has_policy = self.has_spinn and hasattr(model, 'policy_loss')
        self.has_value = self.has_spinn and hasattr(model, 'value_loss')
        self.has_epsilon = self.has_spinn and hasattr(model.spinn, "epsilon")
        self.has_spinn_temperature = self.has_spinn and hasattr(
            model.spinn, "temperature")
        self.has_pyramid_temperature = hasattr(model, "temperature_to_display")


def inspect(model):
    return InspectModel(model)

def finalStats(final_A, logger):
    # todo: average total candidates and mentions docs,time
    # time_metric = time_per_token(final_A.get('total_candidates'), A.get('total_time'))
    logger.Log("dev best:\n macc:{}, dacc:{}.".format(
                                    final_A.get_avg('dev_macc'), final_A.get_avg('dev_dacc')))
    logger.Log("test best:\n macc:{}, dacc:{}.".format(
                                    final_A.get_avg('test_macc'), final_A.get_avg('test_dacc')))

def stats(model, trainer, A, log_entry):
    time_metric = time_per_token(A.get('total_candidates'), A.get('total_time'))

    log_entry.step = trainer.step
    log_entry.mention_accuracy = A.get_avg('mention_acc')
    log_entry.document_accuracy = A.get_avg('doc_acc')
    log_entry.total_cost = A.get_avg('total_cost')  # not actual mean
    log_entry.learning_rate = trainer.learning_rate
    log_entry.time_per_token_seconds = time_metric

    return log_entry

def eval_stats(model, A, eval_data):
    mention_correct = A.get('mention_correct')
    batch_mentions = A.get('mention_batch')
    eval_data.eval_mention_accuracy = sum(mention_correct) / float(sum(batch_mentions))

    doc_acc_per_batch = A.get('macro_acc')
    batch_docs = A.get('doc_batch')
    eval_data.eval_document_accuracy = sum(doc_acc_per_batch) / float(len(batch_docs))

    time_metric = time_per_token(A.get('total_candidates'), A.get('total_time'))
    eval_data.time_per_token_seconds = time_metric

    return eval_data


def train_format(log_entry):
    stats_str = "Step: {step}"

    # Accuracy Component.
    stats_str += " Acc: mt {ment_acc:.5f} dc {doc_acc:.5f}"

    # Cost Component.
    stats_str += " Cost: to {total_loss:.5f}"

    # Time Component.
    stats_str += " Time: {time:.5f}"

    return stats_str


def eval_format(evaluation):
    eval_str = "Step: {step} Eval acc: mt {ment_acc:.5f} dc {doc_acc:.5f} Time: {time:.5f}"

    return eval_str

def log_formatter(log_entry):
    """Defines the log string to print to std error."""
    args = {
        'step': log_entry.step,
        'ment_acc': log_entry.mention_accuracy,
        'doc_acc': log_entry.document_accuracy,
        'total_loss': log_entry.total_cost,
        'time': log_entry.time_per_token_seconds,
    }

    log_str = train_format(log_entry).format(**args)
    if len(log_entry.evaluation) > 0:
        for evaluation in log_entry.evaluation:
            eval_args = {
                'step': log_entry.step,
                'ment_acc': evaluation.eval_mention_accuracy,
                'doc_acc': evaluation.eval_document_accuracy,
                'time': evaluation.time_per_token_seconds,
            }
            log_str += '\n' + \
                eval_format(evaluation).format(**eval_args)

    return log_str


def create_log_formatter():
    def fmt(log_entry):
        return log_formatter(log_entry)
    return fmt

# --- Sample printing ---

def print_samples(output, vocabulary, docs, only_one=False):
    word_vocab, entity_vocab, sense_vocab, id2wiki_vocab = vocabulary

    ent_label_vocab = dict(
            [(entity_vocab[id], id2wiki_vocab[id]) for id in entity_vocab if id in id2wiki_vocab])
    ent_label_vocab[0] = 'PAD'
    ent_label_vocab[1] = 'UNK'
    ent_id_vocab = dict([(entity_vocab[key], key) for key in entity_vocab])
    word_label_vocab = dict(
        [(word_vocab[key], key) for key in word_vocab if key in word_vocab])

    sample_sequences = []
    m_idx = -1
    for doc in docs:
        doc_sequence = [[word_label_vocab[token] for token in sent] for sent in doc.sentences]
        for i, m in enumerate(doc.mentions):
            m_idx += 1
            cand_size = len(m.candidates)
            doc_sequence[m._sent_idx][m._pos_in_sent] = "[[ " + doc_sequence[m._sent_idx][m._pos_in_sent]
            doc_sequence[m._sent_idx][m._pos_in_sent+m._mention_length-1] += " ]]"

            out_mention = output[m_idx, :]
            pred_cid = m.candidates[np.argmax(out_mention)].id
            doc_sequence.append(["mention:", str(i), "correct:", "true" if m.gold_ent_id()==pred_cid else "False",
                                     ",cand:", str(cand_size),
                                     ", gold:", ent_id_vocab[m.gold_ent_id()], m.gold_ent_str()])

            for j, p in enumerate(out_mention):
                if j < cand_size:
                    doc_sequence.append(["candidate", str(j), ":", ent_id_vocab[m.candidates[j].id], ent_label_vocab[m.candidates[j].id], str(p)])
                else:
                    doc_sequence.append(["pad", str(j), ":", str(p)])
        doc_line = ''
        for line in doc_sequence:
            doc_line += ' '.join(line) + '\n'
        sample_sequences.append(doc_line)
    return sample_sequences