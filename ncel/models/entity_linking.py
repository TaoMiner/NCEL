
import os
import json
import random
import sys
import time
import math

import gflags
import numpy as np

from ncel.utils import afs_safe_logger
from ncel.utils.data import SimpleProgressBar
from ncel.utils.logging import stats, train_accumulate, create_log_formatter
from ncel.utils.logging import eval_stats, eval_accumulate, prettyprint_trees
import ncel.utils.logging_pb2 as pb
from ncel.utils.trainer import ModelTrainer

from ncel.models.base import get_data_manager, get_flags, get_batch
from ncel.models.base import flag_defaults, init_model, log_path
from ncel.models.base import load_data_and_embeddings

FLAGS = gflags.FLAGS

def run(only_forward=False):
    logger = afs_safe_logger.ProtoLogger(
        log_path(FLAGS), print_formatter=create_log_formatter(),
        write_proto=FLAGS.write_proto_to_log)
    header = pb.NcelHeader()

    data_manager = get_data_manager(FLAGS.data_type)

    logger.Log("Flag Values:\n" +
               json.dumps(FLAGS.FlagValuesDict(), indent=4, sort_keys=True))

    # Get Data and Embeddings
    vocabulary, initial_embeddings, training_data_iter, eval_iterators, training_data_length = \
        load_data_and_embeddings(FLAGS, data_manager, logger,
                                 FLAGS.training_data_path, FLAGS.eval_data_path)
    '''
    f = open("./vocab.txt", "w")
    for k in vocabulary:
        f.write("{0}\t{1}\n".format(k, vocabulary[k]))
    f.close()
    '''
    # Build model.
    vocab_size = len(vocabulary)
    num_classes = len(set(data_manager.LABEL_MAP.values()))

    model = init_model(
        FLAGS, logger, initial_embeddings, vocab_size, num_classes, data_manager, header)
    epoch_length = int(training_data_length / FLAGS.batch_size)
    trainer = ModelTrainer(model, logger, epoch_length, vocabulary, FLAGS)

    header.start_step = trainer.step
    header.start_time = int(time.time())

    # Do an evaluation-only run.
    logger.LogHeader(header)  # Start log_entry logging.
    if only_forward:
        log_entry = pb.SpinnEntry()
        for index, eval_set in enumerate(eval_iterators):
            log_entry.Clear()
            evaluate(
                FLAGS,
                model,
                eval_set,
                log_entry,
                logger,
                trainer,
                vocabulary,
                show_sample=True,
                eval_index=index)
            print(log_entry)
            logger.LogEntry(log_entry)
    else:
        train_loop(
            FLAGS,
            model,
            trainer,
            training_data_iter,
            eval_iterators,
            logger,
            vocabulary)


if __name__ == '__main__':
    get_flags()

    # Parse command line flags.
    FLAGS(sys.argv)

    flag_defaults(FLAGS)

    run(only_forward=FLAGS.expanded_eval_only_mode)