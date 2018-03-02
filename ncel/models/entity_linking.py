# -*- coding: utf-8 -*-
import os
import json
import sys
import time

import gflags

from ncel.utils import afs_safe_logger
from ncel.utils.data import SimpleProgressBar, get_batch
from ncel.utils.logging import stats, create_log_formatter
from ncel.utils.logging import eval_stats, print_samples, finalStats
import ncel.utils.logging_pb2 as pb
from ncel.utils.trainer import ModelTrainer

from ncel.models.base import get_flags
from ncel.models.base import init_model, log_path, flag_defaults
from ncel.models.base import load_data_and_embeddings

from ncel.utils.misc import Accumulator, EvalReporter, ComputeAccuracy, inspectBatch

from ncel.utils.layers import to_gpu

from ncel.utils.Candidates import getCandidateHandler

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable

FLAGS = gflags.FLAGS

def evaluate(FLAGS, model, eval_set, log_entry,
             logger, vocabulary=None, show_sample=False, eval_index=0, report_sample=False):
    dataset = eval_set

    A = Accumulator()
    eval_log = log_entry.evaluation.add()
    reporter = EvalReporter()

    # Evaluate
    total_batches = len(dataset)
    progress_bar = SimpleProgressBar(
        msg="Run Eval",
        bar_length=60,
        enabled=FLAGS.show_progress_bar)
    progress_bar.step(0, total=total_batches)
    total_candidates = 0
    start = time.time()
    samples = None

    model.eval()

    for i, dataset_batch in enumerate(dataset):
        base, context1, context2, m_strs, cids, cids_sense, num_candidates, num_mentions, y = get_batch(dataset_batch,
                                FLAGS.local_context_window, use_lr_context=FLAGS.use_lr_context,
                                split_by_sent=FLAGS.split_by_sent)
        # batch_candidates = eval_num_candidates_batch.sum()
        # Run model. output: batch_size * node_num
        output = model(context1, base, cids, m_strs,
                       contexts2=context2, candidates_sense=cids_sense,
                       num_mentions=num_mentions, length=num_candidates)

        if show_sample:
            samples = print_samples(output.data.cpu().numpy(), vocabulary, dataset_batch, only_one=True)
            show_sample=False

        # Calculate accuracy.
        target = torch.from_numpy(y).long()
        batch_mentions, mention_correct, batch_docs, doc_acc_per_batch =\
            ComputeAccuracy(output, target, dataset_batch)

        A.add('mention_correct', mention_correct)
        A.add('mention_batch', batch_mentions)
        A.add('macro_acc', doc_acc_per_batch)
        A.add('doc_batch', batch_docs)

        # Update Aggregate Accuracies
        batch_candidates = num_candidates.sum()
        total_candidates += batch_candidates

        if FLAGS.write_eval_report and report_sample:
            batch_samples = print_samples(output.data.cpu().numpy(), vocabulary, dataset_batch, only_one=False)
            if FLAGS.model_type in ["PNCEL"] :
                # log graph
                _, entity_vocab, _, id2wiki_vocab = vocabulary
                docs_edges = model.getGraphSample(cids, num_mentions, entity_vocab, id2wiki_vocab, only_one=False)
                for i, edges in enumerate(docs_edges):
                    batch_samples[i] += edges
            reporter.save_batch(batch_samples)

        # Print Progress
        progress_bar.step(i + 1, total=total_batches)
    progress_bar.finish()
    if samples is not None:
        logger.Log('Sample: ' + samples[0])

    end = time.time()
    total_time = end - start

    A.add('total_candidates', total_candidates)
    A.add('total_time', total_time)

    eval_stats(model, A, eval_log)

    if FLAGS.write_eval_report and report_sample:
        eval_report_path = os.path.join(
            FLAGS.log_path,
            FLAGS.experiment_name +
            ".eval_set_" +
            str(eval_index) +
            ".report")
        reporter.write_report(eval_report_path)

    eval_mention_accuracy = eval_log.eval_mention_accuracy
    eval_document_accuracy = eval_log.eval_document_accuracy

    return eval_mention_accuracy, eval_document_accuracy

# length: batch_size
def sequence_mask(sequence_length, max_length):
    lengths = torch.from_numpy(sequence_length).long()
    batch_size = lengths.size()[0]
    seq_range = torch.arange(0, max_length).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length)
    seq_length_expand = lengths.unsqueeze(1)
    return seq_range_expand < seq_length_expand

def train_loop(
        FLAGS,
        model,
        trainer,
        training_data_iter,
        eval_iterators,
        logger,
        vocabulary,
        final_A):
    # Accumulate useful statistics.
    A = Accumulator(maxlen=FLAGS.deque_length)

    # Train.
    logger.Log("Training.")

    # New Training Loop
    progress_bar = SimpleProgressBar(
        msg="Training",
        bar_length=60,
        enabled=FLAGS.show_progress_bar)
    progress_bar.step(i=0, total=FLAGS.statistics_interval_steps)

    log_entry = pb.NcelEntry()
    for _ in range(trainer.step, FLAGS.training_steps):
        if (trainer.step - trainer.best_dev_step) > FLAGS.early_stopping_steps_to_wait:
            logger.Log('No improvement after ' +
                       str(FLAGS.early_stopping_steps_to_wait) +
                       ' steps. Stopping training.')
            break

        # set model in training mode
        model.train()

        log_entry.Clear()
        log_entry.step = trainer.step
        should_log = False

        start = time.time()
        doc_batch = next(training_data_iter)
        batch = get_batch(doc_batch, FLAGS.local_context_window, use_lr_context=FLAGS.use_lr_context,
                                    split_by_sent=FLAGS.split_by_sent)
        base, context1, context2, m_strs, cids, cids_sense, num_candidates, num_mentions, y = batch
        # check training data
        # inspectBatch(batch, vocabulary, doc_batch)

        total_candidates = num_candidates.sum()

        # Reset cached gradients.
        trainer.optimizer_zero_grad()

        # Run model. output: batch_size * max_cand_num
        output = model(context1, base, cids, m_strs,
                       contexts2=context2, candidates_sense=cids_sense,
                       num_mentions=num_mentions, length=num_candidates)

        target = torch.from_numpy(y).long()
        # Calculate accuracy.
        batch_mentions, mention_correct, batch_docs, doc_acc_per_batch = \
            ComputeAccuracy(output, target, doc_batch)

        # Calculate loss.
        loss = nn.CrossEntropyLoss()(output, to_gpu(Variable(target, volatile=False)))
        # bce_loss = nn.BCELoss()(output.masked_select(vmask2d), to_gpu(Variable(target, volatile=False)))
        # for n,p in model.named_parameters():
        #   print('===========\nbefore gradient:{}\n----------\n{}'.format(n, p.grad))
        # Backward pass.
        loss.backward()
        # for n,p in model.named_parameters():
        #     print('===========\nbefore gradient:{}\n----------\n{}'.format(n, p.grad))
        # Hard Gradient Clipping
        nn.utils.clip_grad_norm([param for name, param in model.named_parameters() if name not in
                                 ["word_embed.embed.weight", "entity_embed.embed.weight",
                                  "sense_embed.embed.weight", "mu_embed.embed.weight"]],
                                FLAGS.clipping_max_value)

        # Gradient descent step.
        trainer.optimizer_step()

        end = time.time()

        total_time = end - start

        A.add('mention_acc', mention_correct/float(batch_mentions))
        A.add('doc_acc', doc_acc_per_batch)
        A.add('total_candidates', total_candidates)
        A.add('total_time', total_time)

        if trainer.step % FLAGS.statistics_interval_steps == 0:
            A.add('total_cost', loss.data[0])
            stats(model, trainer, A, log_entry)
            should_log = True
            progress_bar.finish()

        if trainer.step > 0 and trainer.step % FLAGS.eval_interval_steps == 0:
            should_log = True
            # note: at most tow eval set due to training recording best
            test_acc = None
            for index, eval_set in enumerate(eval_iterators):
                acc = evaluate(
                    FLAGS, model, eval_set, log_entry, logger, show_sample=FLAGS.show_sample, vocabulary=vocabulary, eval_index=index)
                if index == 0: dev_acc = acc
                else: test_acc = acc
            trainer.new_accuracy(dev_acc, test_acc=test_acc)
            progress_bar.reset()

        if trainer.step > FLAGS.ckpt_step and trainer.step % FLAGS.ckpt_interval_steps == 0:
            should_log = True
            trainer.checkpoint()

        if should_log:
            logger.LogEntry(log_entry)

        progress_bar.step(i=(trainer.step % FLAGS.statistics_interval_steps) + 1,
                          total=FLAGS.statistics_interval_steps)
    # record train acc and eval acc
    final_A.add('dev_macc', trainer.best_dev_macc)
    final_A.add('dev_dacc', trainer.best_dev_dacc)
    final_A.add('test_macc', trainer.best_test_macc)
    final_A.add('test_dacc', trainer.best_test_dacc)

def run(only_forward=False):
    # todo : revise create_log_formatter
    logger = afs_safe_logger.ProtoLogger(
        log_path(FLAGS), print_formatter=create_log_formatter(),
        write_proto=FLAGS.write_proto_to_log)
    header = pb.NcelHeader()

    candidate_handler = getCandidateHandler()

    logger.Log("Flag Values:\n" + json.dumps(FLAGS.FlagValuesDict(), indent=4, sort_keys=True))

    # Get Data and Embeddings
    vocabulary, initial_embeddings, training_data_iter, eval_iterators,\
        training_data_length, base_feature_dim = load_data_and_embeddings(FLAGS, logger, candidate_handler)

    word_embeddings, entity_embeddings, sense_embeddings, mu_embeddings = initial_embeddings

    # Build model.

    model = init_model(FLAGS, base_feature_dim, initial_embeddings, logger, header)
    epoch_length = int(training_data_length / FLAGS.batch_size)
    trainer = ModelTrainer(model, logger, epoch_length, vocabulary, FLAGS)

    header.start_step = trainer.step
    header.start_time = int(time.time())

    # Do an evaluation-only run.
    logger.LogHeader(header)  # Start log_entry logging.
    if only_forward:
        log_entry = pb.NcelEntry()
        for index, eval_set in enumerate(eval_iterators):
            log_entry.Clear()
            evaluate(
                FLAGS,
                model,
                eval_set,
                log_entry,
                logger,
                show_sample=True,
                vocabulary=vocabulary,
                eval_index=index,
                report_sample=True)
            print(log_entry)
            logger.LogEntry(log_entry)
    else:
        final_A = Accumulator(maxlen=FLAGS.deque_length)
        train_loop(
            FLAGS,
            model,
            trainer,
            training_data_iter,
            eval_iterators,
            logger,
            vocabulary,
            final_A)
        finalStats(final_A, logger)

if __name__ == '__main__':
    get_flags()

    # Parse command line flags.
    FLAGS(sys.argv)
    flag_defaults(FLAGS)

    run(only_forward=FLAGS.eval_only_mode)
