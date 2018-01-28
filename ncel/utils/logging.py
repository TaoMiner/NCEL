# -*- coding: utf-8 -*-
"""
logging.py

Log format convenience methods for training spinn.

"""

from ncel.utils.misc import time_per_token


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
    logger.Log("dev best:\n cacc:{}, macc:{}, dacc:{}.".format(
        final_A.get_avg('dev_cacc'), final_A.get_avg('dev_macc'), final_A.get_avg('dev_dacc')))
    # todo: check exist and revise dev or test string
    logger.Log("test best:\n cacc:{}, macc:{}, dacc:{}.".format(
        final_A.get_avg('test_cacc'), final_A.get_avg('test_macc'), final_A.get_avg('test_dacc')))

def stats(model, trainer, A, log_entry):
    time_metric = time_per_token(A.get('total_candidates'), A.get('total_time'))

    log_entry.step = trainer.step
    log_entry.candidate_accuracy = A.get_avg('candidate_acc')
    log_entry.mention_accuracy = A.get_avg('mention_acc')
    log_entry.document_accuracy = A.get_avg('doc_acc')
    log_entry.total_cost = A.get_avg('total_cost')  # not actual mean
    log_entry.learning_rate = trainer.learning_rate
    log_entry.time_per_token_seconds = time_metric

    return log_entry

def eval_stats(model, A, eval_data):
    candidate_correct = A.get('candidate_correct')
    batch_candidates = A.get('candidate_batch')
    eval_data.eval_candidate_accuracy = sum(candidate_correct) / float(sum(batch_candidates))

    mention_correct = A.get('mention_correct')
    batch_mentions = A.get('mention_batch')
    eval_data.eval_mention_accuracy = sum(mention_correct) / float(sum(batch_mentions))

    doc_acc_per_batch = A.get('macro_acc')
    batch_docs = A.get('doc_batch')
    eval_data.eval_document_accuracy = sum(doc_acc_per_batch) / float(len(batch_docs))

    time_metric = time_per_token(A.get('total_tokens'), A.get('total_time'))
    eval_data.time_per_token_seconds = time_metric

    return eval_data


def train_format(log_entry):
    stats_str = "Step: {step}"

    # Accuracy Component.
    stats_str += " Acc: cd {cand_acc:.5f} mt {ment_acc:.5f} dc {doc_acc:.5f}"

    # Cost Component.
    stats_str += " Cost: to {total_loss:.5f}"

    # Time Component.
    stats_str += " Time: {time:.5f}"

    return stats_str


def eval_format(evaluation):
    eval_str = "Step: {step} Eval acc: cd {cand_acc:.5f} mt {ment_acc:.5f} dc {doc_acc:.5f} Time: {time:.5f}"

    return eval_str

def log_formatter(log_entry):
    """Defines the log string to print to std error."""
    args = {
        'step': log_entry.step,
        'cand_acc': log_entry.candidate_accuracy,
        'ment_acc': log_entry.mention_accuracy,
        'doc_acc': log_entry.document_accuracy,
        'total_loss': log_entry.total_cost,
    }

    log_str = train_format(log_entry).format(**args)
    if len(log_entry.evaluation) > 0:
        for evaluation in log_entry.evaluation:
            eval_args = {
                'step': log_entry.step,
        'cand_acc': log_entry.candidate_accuracy,
        'ment_acc': log_entry.mention_accuracy,
        'doc_acc': log_entry.document_accuracy,
            }
            log_str += '\n' + \
                eval_format(evaluation).format(**eval_args)

    return log_str


def create_log_formatter():
    def fmt(log_entry):
        return log_formatter(log_entry)
    return fmt

# --- Sample printing ---

def print_samples(candidate_ids, output, vocabulary, docs, only_one=False):
    # TODO: Don't show padding.
    word_vocab, entity_vocab, id2wiki_vocab = vocabulary

    ent_label_vocab = dict(
            [(entity_vocab[id], id2wiki_vocab[id]) for id in entity_vocab if id in id2wiki_vocab])
    ent_label_vocab[0] = 'NIL'
    word_label_vocab = dict(
        [(word_vocab[key], key) for key in word_vocab if key in word_vocab])

    sample_sequences = []
    batch_size, max_candidates = candidate_ids.shape
    for b in (list(range(batch_size)) if not only_one else [0]):
        doc_token_sequence = []
        doc = docs[b]
        doc_cand_ids = candidate_ids[b]
        c_idx = 0
        start = 0
        for mention in doc.mentions:
            end = mention._mention_start
            doc_token_sequence.extend([word_label_vocab[token]
                                       for token in doc.tokens[start:end]])
            pred_cid = 0
            largest_prob = 0
            for candidate in mention.candidates:
                cid = doc_cand_ids[c_idx]
                assert candidate.id == cid, "Error match candidates for printing sample!"
                prob_cid = output[b*max_candidates+c_idx, 0]
                if pred_cid > largest_prob : pred_cid = cid
                c_idx += 1
            doc_token_sequence.append("[[{}({})|{}|{}]]".format(ent_label_vocab[pred_cid], largest_prob,
                       mention.gold_ent_str() if not mention._is_NIL else 'NIL', mention._mention_str))
            start = end
        doc_token_sequence.extend([word_label_vocab[token] for token in doc.tokens[start:]])
        sample_sequences.append(' '.join(doc_token_sequence))
    return sample_sequences