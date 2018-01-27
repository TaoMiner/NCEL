# -*- coding: utf-8 -*-
import sys
import os
import time

import gflags

from ncel.data import load_conll_data, load_kbp_data, load_wned_data, load_xlwiki_data
from ncel.utils.data import BuildVocabulary, BuildEntityVocabulary, AddCandidatesToDocs
from ncel.utils.data import PreprocessDataset, LoadEmbeddingsFromBinary
from ncel.utils.data import MakeEvalIterator, MakeTrainingIterator, MakeCrossIterator
from ncel.models.featureGenerator import *
from ncel.utils.logparse import parse_flags

import ncel.models.ncel as ncel

from functools import reduce

import torch

FLAGS = gflags.FLAGS

def log_path(FLAGS, load=False):
    lp = FLAGS.load_log_path if load else FLAGS.log_path
    en = FLAGS.load_experiment_name if load else FLAGS.experiment_name
    return os.path.join(lp, en) + ".log"

def get_batch(batch):
    # x: batch_size * max_candidates * feature_dim
    # candidate_ids, y: batch_size * max_candidates
    # num_candidates: batch_size
    # docs: batch_size
    x, candidate_ids, y, num_candidates, docs = batch

    max_length = np.max(num_candidates)

    # Truncate batch.
    x_batch = truncate(x, max_length, False)
    candidate_ids_batch = truncate(candidate_ids, max_length, True)
    y_batch = truncate(y, max_length, False)

    return x_batch, candidate_ids_batch, y_batch, num_candidates, docs

def truncate(data, max_length, is_c):
    if not is_c:
        data = data[:, :max_length, :]
    else:
        data = data[:, :max_length]
    return data

def get_data_manager(data_type):
    # Select data format.
    if data_type == "conll":
        data_manager = load_conll_data
    elif data_type in ["kbp10", "kbp15", "kbp16"]:
        data_manager = load_kbp_data
    elif data_type in ["msnbc", "aquaint", "ace04", "wiki13", "clueweb12"]:
        data_manager = load_wned_data
    elif data_type == "xlwiki":
        data_manager = load_xlwiki_data
    else:
        raise NotImplementedError

    return data_manager

def get_feature_manager(embeddings, embedding_dim,
                 str_sim=True, prior=True, hasAtt=True,
                 local_context_window=5, global_context_window=5):

    return FeatureGenerator(embeddings, embedding_dim,
                     str_sim=str_sim, prior=prior, hasAtt=hasAtt,
                 local_context_window=local_context_window,
                global_context_window=global_context_window)

def load_data_and_embeddings(
        FLAGS,
        data_manager,
        logger,
        candidate_manager):

    # dev, train and eval
    # conll only one file, controlled by genre (FLAG.test_genre)

    # no dev, train and eval
    # xlwiki only text path of training or eval, controlled by genre (FLAG.genre)

    # kbp15,16 may contain multiple training and eval [path, file]
    # kbp10 only one training and eval [path, file]

    # must cross_validation
    # wned only one eval [path, file]
    raw_training_data = None
    if not FLAGS.eval_only_mode and FLAGS.cross_validation <= 0 :
        if FLAGS.data_type in ["kbp10", "kbp15", "kbp16"]:
            raw_training_data = []
            text_paths = FLAGS.training_text_path.split(':')
            mention_files = FLAGS.training_mention_file.split(':')
            assert len(text_paths) == len(mention_files), "Error training data path!"

            for i, path in enumerate(text_paths):
                raw_training_data.extend(data_manager.load_data(
                    text_path=path, mention_file=mention_files[i],
                    kbp_id2wikiid_file=FLAGS.kbp2wikiId_file,
                    include_unresolved=FLAGS.include_unresolved, lowercase=FLAGS.lowercase,
                    wiki_entity_file=FLAGS.wiki_entity_vocab))
        else:
            genre = 2 if FLAGS.data_type == "conll" else FLAGS.genre
            raw_training_data = data_manager.load_data(
                text_path=FLAGS.training_text_path, mention_file=FLAGS.training_mention_file,
                genre=genre, include_unresolved=FLAGS.include_unresolved, lowercase=FLAGS.lowercase)

    raw_eval_sets = []
    if FLAGS.data_type == "conll":
        genre = [0, 1, 2] if FLAGS.genre == 2 else ([0, 1] if FLAGS.genre == 1 else [1, 0])
        for i in genre:
            raw_eval_sets.append(data_manager.load_data(
                mention_file=FLAGS.training_mention_file, genre=i,
                include_unresolved=FLAGS.include_unresolved,
                lowercase=FLAGS.lowercase))
    elif FLAGS.data_type == "xlwiki":
        raw_eval_sets.append(data_manager.load_data(
            text_path=FLAGS.eval_text_path, genre=FLAGS.genre,
            include_unresolved=FLAGS.include_unresolved, lowercase=FLAGS.lowercase,
            wiki_entity_file=FLAGS.wiki_entity_vocab))
    else:
        text_paths = FLAGS.eval_text_path.split(':')
        mention_files = FLAGS.eval_mention_file.split(':')
        assert len(text_paths) == len(mention_files), "Error eval data path!"

        for i, path in enumerate(text_paths):
            raw_eval_sets.append(data_manager.load_data(
                text_path=path, mention_file=mention_files[i],
                kbp_id2wikiid_file=FLAGS.kbp2wikiId_file,
                include_unresolved=FLAGS.include_unresolved, lowercase=FLAGS.lowercase,
                wiki_entity_file=FLAGS.wiki_entity_vocab))

    # Prepare the word and mention vocabulary.
    word_vocab, mention_vocab = BuildVocabulary(
        raw_training_data,
        raw_eval_sets,
        FLAGS.word_embedding_file,
        logger=logger)

    candidate_handler = candidate_manager(FLAGS.candidates_file, vocab=mention_vocab, lowercase=FLAGS.lowercase)
    candidate_handler.loadCandiates()
    logger.Log("Load "+ str(candidate_handler._candidates_total) + " candidates for "
               + str(len(candidate_handler._mention_dict)) + " mention types!")
    candidate_handler.loadPrior(FLAGS.entity_prior_file)
    id2wiki_vocab = candidate_handler.loadWikiid2Label(FLAGS.wiki_entity_vocab,
                                       id_vocab=candidate_handler._candidate_entities)

    entity_vocab = BuildEntityVocabulary(candidate_handler._candidate_entities,
                                         FLAGS.entity_embedding_file, FLAGS.sense_embedding_file,
                                            logger=logger)

    # Load pretrained embeddings.
    logger.Log("Loading vocabulary with " + str(len(word_vocab))
               + " words from " + FLAGS.word_embedding_file)
    word_embeddings = LoadEmbeddingsFromBinary(
        word_vocab, FLAGS.embedding_dim, FLAGS.word_embedding_file)

    logger.Log("Loading vocabulary with " + str(len(entity_vocab))
               + " entities and senses from " + FLAGS.entity_embedding_file
               + ", and " + FLAGS.sense_embedding_file)
    entity_embeddings = LoadEmbeddingsFromBinary(
        entity_vocab, FLAGS.embedding_dim, FLAGS.entity_embedding_file)

    sense_embeddings, mu_embeddings = LoadEmbeddingsFromBinary(
        entity_vocab, FLAGS.embedding_dim, FLAGS.sense_embedding_file, isSense=True)

    initial_embeddings = (word_embeddings, entity_embeddings, sense_embeddings, mu_embeddings)

    feature_manager = get_feature_manager(initial_embeddings, FLAGS.embedding_dim,
                 str_sim=FLAGS.str_sim, prior=FLAGS.prior, hasAtt=FLAGS.att,
                 local_context_window=FLAGS.local_context_window,
                  global_context_window=FLAGS.global_context_window)
    # update mention candidates
    topn_candidates = FLAGS.max_candidates_per_document if FLAGS.max_candidates_per_document > 0 else None

    # Trim dataset, convert token sequences to integer sequences, crop, and
    # pad. construct data iterator
    logger.Log("Preprocessing data.")
    eval_sets = []
    for i, raw_eval_data in enumerate(raw_eval_sets):
        AddCandidatesToDocs(raw_eval_sets[i], candidate_handler,
                            vocab=entity_vocab, topn=topn_candidates,
                            include_unresolved=FLAGS.include_unresolved)
        eval_data = PreprocessDataset(raw_eval_sets[i],
                                      word_vocab,
                                      entity_vocab,
                                      FLAGS.seq_length,
                                      FLAGS.doc_length,
                                      FLAGS.max_candidates_per_document,
                                      feature_manager,
                                      logger=logger,
                                      include_unresolved=FLAGS.include_unresolved,
                                      allow_cropping=FLAGS.allow_cropping)
        eval_sets.append(eval_data)
    training_data_iter = None
    training_data_length = 0
    if raw_training_data is not None:
        AddCandidatesToDocs(raw_training_data, candidate_handler,
                            vocab=entity_vocab, topn=topn_candidates,
                            include_unresolved=FLAGS.include_unresolved)
        training_data = PreprocessDataset(raw_training_data,
                                          word_vocab,
                                          entity_vocab,
                                          FLAGS.seq_length,
                                          FLAGS.doc_length,
                                          FLAGS.max_candidates_per_document,
                                          feature_manager,
                                          logger=logger,
                                          include_unresolved=FLAGS.include_unresolved,
                                          allow_cropping=FLAGS.allow_cropping)
        training_data_length = training_data[0].shape[0]
        training_data_iter = []
        training_data_iter.append(MakeTrainingIterator(training_data, FLAGS.batch_size, FLAGS.smart_batching))

    if FLAGS.cross_validation > 0:
        logger.Log("Creating cross validation batch iterators from eval data alone ...")
        # get both train_iter and eval_iter from eval data according to cur_validation
        training_data_iter, eval_iterators, training_data_length = MakeCrossIterator(eval_sets[0],
                                                               FLAGS.batch_size,
                                                               FLAGS.cross_validation+2)
    else:
        eval_iterators = []
        # get eval_iter from eval data
        eval_itset = []
        for eval_data in eval_sets:
            eval_it = MakeEvalIterator(
                eval_data,
                FLAGS.batch_size)
            eval_itset.append(eval_it)
        eval_iterators.append(eval_itset)

    feature_dim = feature_manager.getFeatureDim()

    vocabulary = (word_vocab, entity_vocab, id2wiki_vocab)

    return vocabulary, initial_embeddings, training_data_iter, eval_iterators, training_data_length, feature_dim


# python entity_linking.py -log_path -experiment_name -cross_validation -data_type -genre
def get_flags():
    # Debug settings.
    gflags.DEFINE_bool(
        "debug",
        False,
        "Set to True to disable debug_mode and type_checking.")
    gflags.DEFINE_bool(
        "show_progress_bar",
        True,
        "Turn this off when running experiments on HPC.")
    gflags.DEFINE_integer(
        "deque_length",
        100,
        "Max trailing examples to use when computing average training statistics.")
    gflags.DEFINE_string("experiment_name", "", "")
    gflags.DEFINE_string("load_experiment_name", None, "")
    gflags.DEFINE_string(
        "log_path",
        "./logs",
        "A directory in which to write logs.")
    gflags.DEFINE_string(
        "load_log_path",
        None,
        "A directory from which to read logs.")
    gflags.DEFINE_boolean(
        "write_proto_to_log",
        False,
        "Write logs in a protocol buffer format.")
    gflags.DEFINE_string(
        "ckpt_path", None, "Where to save/load checkpoints. Can be either "
                           "a filename or a directory. In the latter case, the experiment name serves as the "
                           "base for the filename.")
    gflags.DEFINE_integer(
        "ckpt_step",
        1000,
        "Steps to run before considering saving checkpoint.")
    gflags.DEFINE_boolean(
        "ckpt_on_best_dev_error",
        True,
        "If error on the first eval set (the dev set) is "
        "at most 0.99 of error at the previous checkpoint, save a special 'best' checkpoint.")
    gflags.DEFINE_boolean(
        "load_best",
        False,
        "If True, attempt to load 'best' checkpoint.")
    gflags.DEFINE_integer("ckpt_interval_steps", 5000,
                          "Update the checkpoint on disk at this interval.")

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

    # Data settings.
    gflags.DEFINE_string("training_text_path", None, "Can contain multiple file paths, separated "
                                "using ':' tokens.")
    gflags.DEFINE_string("training_mention_file", None, "Can contain multiple file paths according "
                                                        "to training_text_path, separated using ':' tokens.")
    gflags.DEFINE_string(
        "eval_text_path", None, "Can contain multiple file paths, separated "
                                "using ':' tokens. The first file should be the dev set, and is used for determining "
                                "when to save the early stopping 'best' checkpoints.")
    gflags.DEFINE_string(
        "eval_mention_file", None, "Can contain multiple files according to eval_text_path, separated "
                                "using ':' tokens.")
    gflags.DEFINE_string(
        "candidates_file", None, "Each line contains mention-entities pair, separated by tab.")
    gflags.DEFINE_string(
        "entity_prior_file", None, "line: enti_id tab gobal_prior tab cand_ment::=count tab ...")
    gflags.DEFINE_string("wiki_entity_vocab", None, "line: entity_label \t entity_id")
    gflags.DEFINE_string("word_embedding_file", None, "")
    gflags.DEFINE_string("entity_embedding_file", None, "")
    gflags.DEFINE_string("sense_embedding_file", None, "")

    gflags.DEFINE_boolean(
        "allow_cropping",
        False,
        "Trim overly long training examples to fit. If not set, skip them.")
    gflags.DEFINE_integer("seq_length", 200, "")
    gflags.DEFINE_integer("doc_length", 100, "")
    gflags.DEFINE_integer("max_candidates_per_document", 200, "")
    gflags.DEFINE_integer("topn_candidate", 20, "Use all candidates if set 0.")
    gflags.DEFINE_integer("genre", 1, "For conll, [0:testa, 1:testb, 2:all] for evaluation, "
                                      "For xlwiki, [0:easy, 1:hard, 2:all].")

    # KBP data
    gflags.DEFINE_string(
        "kbp2wikiId_file", None, "Each line: 'kbp_ent_id','\t','wiki_id'.")

    # Optimization settings.
    gflags.DEFINE_enum("optimizer_type", "Adam", ["Adam", "SGD"], "")
    gflags.DEFINE_integer(
        "training_steps",
        10000,
        "Stop training after this point.")
    gflags.DEFINE_float("learning_rate", 0.5, "Used in optimizer.")
    gflags.DEFINE_float("learning_rate_decay_when_no_progress", 0.5,
                        "Used in optimizer. Decay the LR by this much every epoch steps if a new best has not been set in the last epoch.")
    gflags.DEFINE_float("clipping_max_value", 5.0, "")
    gflags.DEFINE_float("l2_lambda", 1e-5, "")
    gflags.DEFINE_float("dropout", 0.1, "Used for dropout.")
    gflags.DEFINE_integer("batch_size", 32, "Minibatch size.")
    gflags.DEFINE_boolean( "smart_batching", True, "Organize batches using sequence length.")
    gflags.DEFINE_boolean("fine_tune_loaded_embeddings", False,
                          "If set, backpropagate into embeddings even when initializing from pretrained.")

    # Evaluation settings
    gflags.DEFINE_boolean("lowercase", True, "When True, ignore case.")
    # todo: cannot process NIL since there are no NIL embeddings, and output y
    gflags.DEFINE_boolean("include_unresolved", False, "When True, include NIL entity.")
    gflags.DEFINE_boolean(
        "eval_only_mode",
        False,
        "If set, a checkpoint is loaded and a forward pass is done to get the predicted candidates."
        "Requirements: Must specify checkpoint path.")
    gflags.DEFINE_boolean(
        "eval_only_mode_use_best_checkpoint",
        True,
        "When in eval_only_mode, load the ckpt_best checkpoint.")
    gflags.DEFINE_integer(
        "cross_validation",
        -1,
        "how many shares for training, plus extra one for dev, one for test. "
        "default -1 indicates no cross_validation."
        "set cross_validation will only take training paths and files.")

    # Model architecture settings.
    gflags.DEFINE_enum(
        "model_type", "NCEL", [
            "GBDT",
            "MLP",
            "NCEL",
            "NCEL_P"], "")
    gflags.DEFINE_boolean("str_sim", True, ".")
    gflags.DEFINE_boolean("prior", True, ".")
    gflags.DEFINE_boolean("att", True, ".")
    gflags.DEFINE_integer("local_context_window", 0, "num of tokens used as contexts, "
                                              "-1 indicates no local context feature, "
                                              "0 indicates the whole sentence. ")
    gflags.DEFINE_integer("global_context_window", 0, ".")

    gflags.DEFINE_integer("embedding_dim", 200, ".")
    gflags.DEFINE_integer(
        "mlp_dim",
        300,
        "Dimension of intermediate MLP layers.")
    gflags.DEFINE_integer("num_mlp_layers", 1, "Number of MLP layers.")
    gflags.DEFINE_boolean(
        "mlp_ln",
        False,
        "When True, layer normalization is used between MLP layers.")
    gflags.DEFINE_integer("gpu", -1, "")

    gflags.DEFINE_integer(
        "gc_dim",
        300,
        "Dimension of intermediate MLP layers.")
    gflags.DEFINE_integer("num_gc_layer", 2, "Number of Graph convolutional layers.")
    gflags.DEFINE_boolean(
        "gc_ln",
        False,
        "When True, layer normalization is used between gc layers.")

    gflags.DEFINE_integer(
        "classifier_dim",
        300,
        "Dimension of output MLP layers.")
    gflags.DEFINE_integer("num_cm_layer", 1, "Number of classifier mlp layers.")
    gflags.DEFINE_boolean(
        "cm_ln",
        False,
        "When True, layer normalization is used between classifier mlp layers.")

    # Display settings.
    gflags.DEFINE_integer(
        "statistics_interval_steps",
        100,
        "Log training set performance statistics at this interval.")
    gflags.DEFINE_integer(
        "eval_interval_steps",
        100,
        "Evaluate at this interval.")
    gflags.DEFINE_integer(
        "sample_interval_steps",
        None,
        "Sample transitions at this interval.")
    gflags.DEFINE_integer(
        "early_stopping_steps_to_wait",
        500,
        "If development set error doesn't improve significantly in this many steps, cease training.")
    gflags.DEFINE_boolean("write_eval_report", False, "")

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

    if not FLAGS.load_log_path:
        FLAGS.load_log_path = FLAGS.log_path

    if not FLAGS.load_experiment_name:
        FLAGS.load_experiment_name = FLAGS.experiment_name

    if not FLAGS.ckpt_path:
        FLAGS.ckpt_path = FLAGS.load_log_path

    if not FLAGS.sample_interval_steps:
        FLAGS.sample_interval_steps = FLAGS.statistics_interval_steps

    if not torch.cuda.is_available():
        FLAGS.gpu = -1

def init_model(
        FLAGS,
        entity_embeddings,
        feature_dim,
        logger,
        logfile_header=None):
    # Choose model.
    logger.Log("Building model.")
    if FLAGS.model_type == "NCEL":
        build_model = ncel.build_model
    else:
        raise NotImplementedError

    model = build_model(entity_embeddings, feature_dim, FLAGS)

    # Debug
    def set_debug(self):
        self.debug = FLAGS.debug
    model.apply(set_debug)

    # Print model size.
    logger.Log("Architecture: {}".format(model))
    if logfile_header:
        logfile_header.model_architecture = str(model)
    total_params = sum([reduce(lambda x, y: x * y, w.size(), 1.0)
                        for w in model.parameters()])
    logger.Log("Total params: {}".format(total_params))
    if logfile_header:
        logfile_header.total_params = int(total_params)

    return model