import sys
import os
import time

import gflags

from ncel.data import load_conll_data
from ncel.utils.logparse import parse_flags

FLAGS = gflags.FLAGS

def log_path(FLAGS, load=False):
    lp = FLAGS.load_log_path if load else FLAGS.log_path
    en = FLAGS.load_experiment_name if load else FLAGS.experiment_name
    return os.path.join(lp, en) + ".log"

def get_data_manager(data_type):
    # Select data format.
    if data_type == "conll":
        data_manager = load_conll_data
    else:
        raise NotImplementedError

    return data_manager

def load_data_and_embeddings(
        FLAGS,
        data_manager,
        logger,
        training_data_path,
        eval_data_path):

    raw_training_data = None
    raw_eval_data = None
    if FLAGS.cross_validation > 0:
        raw_training_data = data_manager.load_data(
            training_data_path, genre='all', lowercase=FLAGS.lowercase)
    elif FLAGS.expanded_eval_only_mode:
        raw_eval_data = data_manager.load_data(
            eval_data_path, genre='all', lowercase=FLAGS.lowercase)
    else:
        raw_training_data = data_manager.load_data(
            training_data_path, genre='all', lowercase=FLAGS.lowercase)
        raw_eval_data = data_manager.load_data(
            eval_data_path, genre='all', lowercase=FLAGS.lowercase)

    if not FLAGS.expanded_eval_only_mode:
        raw_training_data = data_manager.load_data(
            training_data_path, FLAGS.lowercase, eval_mode=False)
    else:
        raw_training_data = None

    raw_eval_sets = []
    for path in eval_data_path.split(':'):
        raw_eval_data = data_manager.load_data(
            path, FLAGS.lowercase, choose_eval, eval_mode=True)
        raw_eval_sets.append((path, raw_eval_data))

    # Prepare the vocabulary.
    if not data_manager.FIXED_VOCABULARY:
        vocabulary = util.BuildVocabulary(
            raw_training_data,
            raw_eval_sets,
            FLAGS.embedding_data_path,
            FLAGS.embedding_format,
            logger=logger,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA)
    else:
        vocabulary = data_manager.FIXED_VOCABULARY
        logger.Log("In fixed vocabulary mode. Training embeddings from scratch.")

    # Load pretrained embeddings.
    if FLAGS.embedding_data_path:
        logger.Log("Loading vocabulary with " + str(len(vocabulary))
                   + " words from " + FLAGS.embedding_data_path)
        initial_embeddings = util.LoadEmbeddingsFromText(
            vocabulary, FLAGS.word_embedding_dim, FLAGS.embedding_data_path) if FLAGS.embedding_format == 't' else util.LoadEmbeddingsFromBinary(
            vocabulary, FLAGS.word_embedding_dim, FLAGS.embedding_data_path)
    else:
        initial_embeddings = None

    # Trim dataset, convert token sequences to integer sequences, crop, and
    # pad.
    logger.Log("Preprocessing training data.")
    if raw_training_data is not None:
        training_data = util.PreprocessDataset(
            raw_training_data,
            vocabulary,
            FLAGS.seq_length,
            data_manager,
            eval_mode=False,
            logger=logger,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
            simple=sequential_only(),
            allow_cropping=FLAGS.allow_cropping,
            pad_from_left=pad_from_left()) if raw_training_data is not None else None
        training_data_iter = util.MakeTrainingIterator(
            training_data, FLAGS.batch_size, FLAGS.smart_batching, FLAGS.use_peano,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA) if raw_training_data is not None else None
        training_data_length = len(training_data[0])
    else:
        training_data_iter = None
        training_data_length = 0

    # Preprocess eval sets.
    eval_iterators = []
    for filename, raw_eval_set in raw_eval_sets:
        logger.Log("Preprocessing eval data: " + filename)
        eval_data = util.PreprocessDataset(
            raw_eval_set, vocabulary,
            FLAGS.eval_seq_length if FLAGS.eval_seq_length is not None else FLAGS.seq_length,
            data_manager, eval_mode=True, logger=logger,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
            simple=sequential_only(),
            allow_cropping=FLAGS.allow_eval_cropping, pad_from_left=pad_from_left())
        eval_it = util.MakeEvalIterator(
            eval_data,
            FLAGS.batch_size,
            FLAGS.eval_data_limit,
            bucket_eval=FLAGS.bucket_eval,
            shuffle=FLAGS.shuffle_eval,
            rseed=FLAGS.shuffle_eval_seed)
        eval_iterators.append((filename, eval_it))

    return vocabulary, initial_embeddings, training_data_iter, eval_iterators, training_data_length

def get_flags():
    # Debug settings.
    gflags.DEFINE_bool(
        "debug",
        False,
        "Set to True to disable debug_mode and type_checking.")
    gflags.DEFINE_string("experiment_name", "", "")
    gflags.DEFINE_string("load_experiment_name", None, "")


    # Data types.
    gflags.DEFINE_enum("data_type",
                       "conll",
                       ["conll",
                        "kbp10",
                        "kbp15",
                        "kbp16",
                        "xlwiki",   # Cross-lingual Wikification Using Multilingual Embeddings
                        # Robust Named Entity Disambiguation with RandomWalks
                        "msnbc",
                        "aquaint",
                        "ace04",
                        "wiki13",
                        "clueweb12"],
                       "el datasets.")

    # logging
    gflags.DEFINE_boolean(
        "write_proto_to_log",
        False,
        "Write logs in a protocol buffer format.")

    # Evaluation settings
    gflags.DEFINE_boolean(
        "expanded_eval_only_mode",
        False,
        "If set, a checkpoint is loaded and a forward pass is done to get the predicted "
        "transitions. The inferred parses are written to the supplied file(s) along with example-"
        "by-example accuracy information. Requirements: Must specify checkpoint path.")
    gflags.DEFINE_integer(
        "cross_validation",
        -1,
        "how many shares for training. default -1 indicates no cross_validation")


def flag_defaults(FLAGS, load_log_flags=False):
    if load_log_flags:
        if FLAGS.load_log_path and os.path.exists(log_path(FLAGS, load=True)):
            log_flags = parse_flags(log_path(FLAGS, load=True))
            for k in list(log_flags.keys()):
                setattr(FLAGS, k, log_flags[k])

            # Optionally override flags from log file.
            FLAGS(sys.argv)

    if not FLAGS.experiment_name:
        timestamp = str(int(time.time()))
        FLAGS.experiment_name = "{}-{}-{}".format(
            FLAGS.data_type,
            FLAGS.model_type,
            timestamp,
        )

    if not FLAGS.git_branch_name:
        FLAGS.git_branch_name = os.popen(
            'git rev-parse --abbrev-ref HEAD').read().strip()

    if not FLAGS.git_sha:
        FLAGS.git_sha = os.popen('git rev-parse HEAD').read().strip()

    if not FLAGS.slurm_job_id:
        FLAGS.slurm_job_id = os.popen('echo $SLURM_JOB_ID').read().strip()

    if not FLAGS.load_log_path:
        FLAGS.load_log_path = FLAGS.log_path

    if not FLAGS.load_experiment_name:
        FLAGS.load_experiment_name = FLAGS.experiment_name

    if not FLAGS.ckpt_path:
        FLAGS.ckpt_path = FLAGS.load_log_path

    if not FLAGS.sample_interval_steps:
        FLAGS.sample_interval_steps = FLAGS.statistics_interval_steps

    if FLAGS.model_type in ["CBOW", "RNN", "ChoiPyramid", "EESC", "LMS"]:
        FLAGS.num_samples = 0

    if FLAGS.model_type == "LMS":
        FLAGS.reduce = "lms"