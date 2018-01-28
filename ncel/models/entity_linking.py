# -*- coding: utf-8 -*-
import os
import json
import sys
import time

import gflags

from ncel.utils import afs_safe_logger
from ncel.utils.data import SimpleProgressBar
from ncel.utils.logging import stats, create_log_formatter
from ncel.utils.logging import eval_stats, print_samples, finalStats
import ncel.utils.logging_pb2 as pb
from ncel.utils.trainer import ModelTrainer

from ncel.models.base import get_data_manager, get_flags, get_batch
from ncel.models.base import init_model, log_path, flag_defaults
from ncel.models.base import load_data_and_embeddings

from ncel.utils.misc import Accumulator, EvalReporter, ComputeMentionAccuracy

from ncel.utils.layers import to_gpu

from ncel.utils.Candidates import getCandidateHandler

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable

FLAGS = gflags.FLAGS

NIL_THRED = 0.1

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
        batch = get_batch(dataset_batch)
        eval_X_batch, eval_adj_batch, eval_y_batch, eval_num_candidates_batch, eval_doc_batch = batch
        batch_candidates = eval_num_candidates_batch.sum()
        # Run model.
        output = model(
            eval_X_batch,
            eval_num_candidates_batch,
            adj=eval_adj_batch)

        if show_sample:
            samples = print_samples(output.data.cpu().numpy(), vocabulary, eval_doc_batch, only_one=True)
            show_sample=False

        # Calculate candidate accuracy.
        target = torch.from_numpy(eval_y_batch).long()

        # get the index of the max log-probability
        pred = output.data.max(2, keepdim=False)[1].cpu()
        total_target = target.size(0)
        candidate_correct = pred.eq(target).sum() - total_target + batch_candidates

        # Calculate mention accuracy.
        batch_mentions, mention_correct, batch_docs, doc_acc_per_batch =\
            ComputeMentionAccuracy(output.data.cpu().numpy(), eval_y_batch, eval_doc_batch, NIL_thred=0.1)

        A.add('candidate_correct', candidate_correct)
        A.add('candidate_batch', batch_candidates)
        A.add('mention_correct', mention_correct)
        A.add('mention_batch', batch_mentions)
        A.add('macro_acc', doc_acc_per_batch)
        A.add('doc_batch', batch_docs)

        # Update Aggregate Accuracies
        total_candidates += batch_candidates

        if FLAGS.write_eval_report and report_sample:
            batch_samples = print_samples(output.data.cpu().numpy(), vocabulary, eval_doc_batch, only_one=False)
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

    eval_candidate_accuracy = eval_log.eval_candidate_accuracy
    eval_mention_accuracy = eval_log.eval_mention_accuracy
    eval_document_accuracy = eval_log.eval_document_accuracy

    return eval_candidate_accuracy, eval_mention_accuracy, eval_document_accuracy

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

        batch = get_batch(next(training_data_iter))
        x, adj, y, num_candidates, docs = batch

        total_candidates = num_candidates.sum()

        # Reset cached gradients.
        trainer.optimizer_zero_grad()

        # Run model. output: batch_size * node_num * 2
        output = model(x, num_candidates, adj=adj)

        # Calculate class accuracy.
        # y: batch_size * node_num
        target = torch.from_numpy(y).long().view(-1)

        # get the index of the max log-probability
        pred = output.data.max(2, keepdim=False)[1].cpu()

        total_target = target.size(0)
        candidate_acc = (pred.eq(target).sum()-total_target+total_candidates) / float(total_candidates)

        # Calculate mention accuracy.
        batch_mentions, mention_correct, batch_docs, doc_acc_per_batch = \
            ComputeMentionAccuracy(output.data.cpu().numpy(), y, docs, NIL_thred=NIL_THRED)

        # Calculate class loss.
        xent_loss = nn.CrossEntropyLoss()(output.view(-1, 2), to_gpu(Variable(target, volatile=False)))

        # Backward pass.
        xent_loss.backward()

        # Hard Gradient Clipping
        nn.utils.clip_grad_norm([param for name, param in model.named_parameters() if name not in ["embed.embed.weight"]], FLAGS.clipping_max_value)

        # Gradient descent step.
        trainer.optimizer_step()

        end = time.time()

        total_time = end - start

        A.add('candidate_acc', candidate_acc)
        A.add('mention_acc', mention_correct/float(batch_mentions))
        A.add('doc_acc', doc_acc_per_batch)
        A.add('total_candidates', total_candidates)
        A.add('total_time', total_time)

        if trainer.step % FLAGS.statistics_interval_steps == 0:
            A.add('total_cost', xent_loss.data[0])
            stats(model, trainer, A, log_entry)
            should_log = True
            progress_bar.finish()

        if trainer.step > 0 and trainer.step % FLAGS.eval_interval_steps == 0:
            should_log = True
            for index, eval_set in enumerate(eval_iterators):
                acc = evaluate(
                    FLAGS, model, eval_set, log_entry, logger, show_sample=(
                        trainer.step %
                        FLAGS.sample_interval_steps == 0), vocabulary=vocabulary, eval_index=index)
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
    final_A.add('dev_cacc', trainer.best_dev_cacc)
    final_A.add('dev_macc', trainer.best_dev_macc)
    final_A.add('dev_dacc', trainer.best_dev_dacc)
    final_A.add('test_cacc', trainer.best_test_cacc)
    final_A.add('test_macc', trainer.best_test_macc)
    final_A.add('test_dacc', trainer.best_test_dacc)

def run(only_forward=False):
    # todo : revise create_log_formatter
    logger = afs_safe_logger.ProtoLogger(
        log_path(FLAGS), print_formatter=create_log_formatter(),
        write_proto=FLAGS.write_proto_to_log)
    header = pb.NcelHeader()

    data_manager = get_data_manager(FLAGS.data_type)

    candidate_handler = getCandidateHandler()

    logger.Log("Flag Values:\n" + json.dumps(FLAGS.FlagValuesDict(), indent=4, sort_keys=True))

    # Get Data and Embeddings
    vocabulary, initial_embeddings, training_data_iter, eval_iterators,\
        training_data_length, feature_dim = load_data_and_embeddings(FLAGS, data_manager, logger, candidate_handler)

    word_embeddings, entity_embeddings, sense_embeddings, mu_embeddings = initial_embeddings

    # Build model.

    model = init_model(FLAGS, entity_embeddings, feature_dim, logger, header)
    epoch_length = int(training_data_length / FLAGS.batch_size)
    trainer = ModelTrainer(model, logger, epoch_length, vocabulary, FLAGS)

    header.start_step = trainer.step
    header.start_time = int(time.time())

    # Do an evaluation-only run.
    logger.LogHeader(header)  # Start log_entry logging.
    total_loops = FLAGS.cross_validation + 2 if FLAGS.cross_validation > 0 else 1

    if only_forward:
        log_entry = pb.NcelEntry()
        for index, eval_set in enumerate(eval_iterators[0]):
            log_entry.Clear()
            evaluate(
                FLAGS,
                model,
                eval_set,
                log_entry,
                logger,
                trainer,
                show_sample=True,
                eval_index=index,
                report_sample=True)
            print(log_entry)
            logger.LogEntry(log_entry)
    else:
        final_A = Accumulator(maxlen=FLAGS.deque_length)
        for cur_loop in range(total_loops):
            logger.Log("Cross validation the {}/{} loop ...".format(cur_loop, total_loops))
            train_loop(
                FLAGS,
                model,
                trainer,
                training_data_iter[cur_loop],
                eval_iterators[cur_loop],
                logger,
                vocabulary,
                final_A)
            finalStats(final_A, logger)
            trainer.reset()
            trainer.optimizer_reset(FLAGS.learning_rate)

if __name__ == '__main__':
    get_flags()

    # Parse command line flags.
    FLAGS(sys.argv)
    flag_defaults(FLAGS)

    run(only_forward=FLAGS.eval_only_mode)