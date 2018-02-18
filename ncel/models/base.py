# -*- coding: utf-8 -*-
import sys
import os
import time

import gflags
import re

from ncel.data import load_conll_data, load_kbp_data, load_wned_data, load_xlwiki_data, load_ncel_data
from ncel.utils.data import BuildVocabulary, BuildEntityVocabulary, AddCandidatesToDocs
from ncel.utils.data import PreprocessDataset, LoadEmbeddingsFromBinary
from ncel.utils.data import MakeEvalIterator, MakeTrainingIterator
from ncel.models.featureGenerator import *
from ncel.utils.logparse import parse_flags
from ncel.utils.model_reader import ModelReader
from ncel.utils.misc import loadWikiVocab, loadRedirectVocab, loadStopWords

import ncel.models.ncel as ncel
import ncel.models.mlp as mlp

from functools import reduce

import torch

FLAGS = gflags.FLAGS

YAMADA_DIM = 300

DATA_TYPE = ["conll",
            "kbp10",
            "kbp15",
            "kbp16",
            "xlwiki",   # Cross-lingual Wikification Using Multilingual Embeddings
            # Robust Named Entity Disambiguation with RandomWalks
            "msnbc",
            "aquaint",
            "ace04",
            "wiki13",
            "clueweb12",
             "ncelwiki"]

def log_path(FLAGS, load=False):
    lp = FLAGS.load_log_path if load else FLAGS.log_path
    en = FLAGS.load_experiment_name if load else FLAGS.experiment_name
    return os.path.join(lp, en) + ".log"

# contexts1 : batch * cand_num * tokens
# contexts2 : batch * cand_num * tokens
# base_feature : batch * cand_num * features
# candidates : batch * cand_num
# candidates_entity: batch * cand_num
# length: batch
# truth: batch
def get_batch(batch, local_window, use_lr_context=True, split_by_sent=True):
    docs = batch
    num_candidates = np.array([len(m.candidates) for doc in docs for m in doc.mentions])
    max_cand_num = max(num_candidates)

    token_pad = 0
    feature_pad = 0.0
    base_feature_dim = -1

    C1 = []
    C2 = []
    B = []
    CID = []
    CID_entity = []
    Y = []

    for doc in docs:
        for m in doc.mentions:
            con1 = []
            con2 = []
            base = []
            cids = []
            cids_entity = []
            # pad tokens
            tmp_con1 = m.left_context(max_len=local_window, split_by_sent=split_by_sent)
            diff = local_window - len(tmp_con1)
            if diff > 0:
                tmp_con1 += [token_pad] * diff

            tmp_con2 = m.right_context(max_len=local_window, split_by_sent=split_by_sent)
            diff = local_window - len(tmp_con2)
            if diff > 0:
                tmp_con2 += [token_pad] * diff

            gold_idx = -1
            for i, c in enumerate(m.candidates):
                base.append(c.getBaseFeature())
                if base_feature_dim <= 0: base_feature_dim = len(c.getBaseFeature())
                con1.append(tmp_con1)
                con2.append(tmp_con2)
                cids.append(c._sense_id)
                cids_entity.append(c.id)
                if c.getIsGlod():
                    gold_idx = i
            assert gold_idx >= 0, "Error: no gold candidate!"
            # padding current mention candidate to max_cand_num
            paddings = max_cand_num - len(m.candidates)
            if paddings > 0:
                base_vec = [feature_pad] * base_feature_dim
                token_vec = [token_pad] * local_window
                for i in range(paddings):
                    base.append(base_vec)
                    con1.append(token_vec)
                    con2.append(token_vec)
                    cids.append(token_pad)
                    cids_entity.append(token_pad)
            B.append(base)
            C1.append(con1)
            C2.append(con2)
            CID.append(cids)
            CID_entity.append(cids_entity)
            Y.append(gold_idx)

    C1 = np.array(C1)
    C2 = np.array(C2)
    if not use_lr_context:
        C1 = np.concatenate((C1, C2), axis=2)
        C2 = None

    return np.array(B), C1, C2, np.array(CID), np.array(CID_entity), num_candidates, np.array(Y)

# todo: cropping over length docs when load data, may be not cropped
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
    elif data_type == "ncelwiki":
        data_manager = load_ncel_data
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

def unwrapDataset(data_tuples):
    datasets = data_tuples.split(",")
    unwraped_data_tuples = []
    for dataset in datasets:
        items = dataset.split(":")
        assert len(items)==4 and len(items[0])>0, "Error: unmatched data tuples!"
        data_type = items[0]
        genre = int(items[1]) if len(items[1])>0 else None
        text_path = items[2] if len(items[2]) > 0 else None
        mention_file = items[3] if len(items[3]) > 0 else None
        unwraped_data_tuples.append([data_type, genre, text_path, mention_file])
    return unwraped_data_tuples

def extractRawData(data_type, text_path, mention_file, genre, FLAGS):
    assert data_type in DATA_TYPE, "Wrong input data types!"
    data_manager = get_data_manager(data_type)
    raw_data = data_manager.load_data(text_path=text_path, mention_file=mention_file,
                kbp_id2wikiid_file=FLAGS.kbp2wikiId_file, genre=genre,
                include_unresolved=FLAGS.include_unresolved, lowercase=FLAGS.lowercase,
                wiki_entity_file=FLAGS.wiki_entity_vocab)

    return raw_data

def load_data_and_embeddings(
        FLAGS,
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
    dataset_types = set()
    raw_training_data = None
    if not FLAGS.eval_only_mode:
        raw_training_data = []
        unwraped_data_tuples = unwrapDataset(FLAGS.training_data)
        for data_tuple in unwraped_data_tuples:
            dataset_types.add(data_tuple[0])
            raw_training_data.extend(extractRawData(data_tuple[0],
                      data_tuple[2], data_tuple[3], data_tuple[1], FLAGS))


    raw_eval_sets = []
    unwraped_data_tuples = unwrapDataset(FLAGS.eval_data)
    for data_tuple in unwraped_data_tuples:
        dataset_types.add(data_tuple[0])
        raw_eval_sets.append(extractRawData(data_tuple[0],
                   data_tuple[2], data_tuple[3], data_tuple[1], FLAGS))

    # replace mention gold id in redirect to entity id
    redirect_vocab = None
    if FLAGS.wiki_redirect_vocab is not None:
        gold_id_set = set()
        if raw_training_data is not None:
            gold_id_set.update([m.gold_ent_id() for doc in raw_training_data
                                for m in doc.mentions if m.gold_ent_id() is not None])
        for eval_data in raw_eval_sets:
            gold_id_set.update([m.gold_ent_id() for doc in eval_data
                                for m in doc.mentions if m.gold_ent_id() is not None])
        redirect_vocab = loadRedirectVocab(FLAGS.wiki_redirect_vocab, id_vocab=gold_id_set)
        if raw_training_data is not None:
            for i, doc in enumerate(raw_training_data):
                for j, mention in enumerate(doc.mentions):
                    if mention.gold_ent_id() in redirect_vocab:
                        raw_training_data[i].mentions[j]._gold_ent_id = redirect_vocab[mention.gold_ent_id()]
        for eval_data in raw_eval_sets:
            for i, doc in enumerate(eval_data):
                for j, mention in enumerate(doc.mentions):
                    if mention.gold_ent_id() in redirect_vocab:
                        eval_data[i].mentions[j]._gold_ent_id = redirect_vocab[mention.gold_ent_id()]

    # Prepare the word and mention vocabulary.
    word_vocab, mention_vocab = BuildVocabulary(
        raw_training_data,
        raw_eval_sets,
        FLAGS.word_embedding_file,
        logger=logger)

    wiki2id_vocab, id2wiki_vocab = loadWikiVocab(FLAGS.wiki_entity_vocab)

    # candidate file types
    candidate_types = []
    files = re.split(r',', FLAGS.candidates_file)
    for f in files:
        tmp_items = re.split(r':', f)
        candidate_types.append(tmp_items[0])

    candidate_handler = candidate_manager(FLAGS.candidates_file, vocab=mention_vocab,
                          lowercase=FLAGS.lowercase, id2label=id2wiki_vocab,
                        label2id=wiki2id_vocab, support_fuzzy=FLAGS.support_fuzzy,
                          redirect_vocab=redirect_vocab, topn=FLAGS.topn_candidate)
    candidate_handler.loadCandidates()
    if FLAGS.save_candidates_path is not None:
        fuzzy_str = 'fuzzy' if FLAGS.support_fuzzy else 'nofuzzy'
        candidate_handler.saveCandidatesToFile(os.path.join(FLAGS.save_candidates_path,
             '-'.join(dataset_types)+'-'.join(candidate_types)+'_candidate_'+fuzzy_str))

    logger.Log("Unk mention types rate: {:2.6f}% ({}/{}), average candidates: {:2.6f}% ({}/{}) from {}!".format(
        (len(mention_vocab)-len(candidate_handler._mention_dict))*100/float(len(mention_vocab)),
        len(mention_vocab) - len(candidate_handler._mention_dict), len(mention_vocab), candidate_handler._candidates_total/float(len(candidate_handler._mention_dict)),
         candidate_handler._candidates_total, len(candidate_handler._mention_dict), FLAGS.candidates_file))

    entity_vocab, sense_vocab = BuildEntityVocabulary(candidate_handler._entity_set,
                                         FLAGS.entity_embedding_file, FLAGS.sense_embedding_file,
                                        logger=logger)

    # Load pretrained embeddings.
    logger.Log("Loading vocabulary with " + str(len(word_vocab))
               + " words from " + FLAGS.word_embedding_file)
    word_embeddings = LoadEmbeddingsFromBinary(
        word_vocab, FLAGS.embedding_dim, FLAGS.word_embedding_file)

    logger.Log("Loading vocabulary with " + str(len(entity_vocab))
               + " entities from " + FLAGS.entity_embedding_file)
    entity_embeddings = LoadEmbeddingsFromBinary(
        entity_vocab, FLAGS.embedding_dim, FLAGS.entity_embedding_file)

    sense_embeddings = None
    mu_embeddings = None
    if sense_vocab is not None:
        sense_embeddings, mu_embeddings = LoadEmbeddingsFromBinary(
            sense_vocab, FLAGS.embedding_dim, FLAGS.sense_embedding_file, isSense=True)
        logger.Log("Loading vocabulary with " + str(len(sense_vocab))
                   + " senses from " + FLAGS.sense_embedding_file)

    initial_embeddings = (word_embeddings, entity_embeddings, sense_embeddings, mu_embeddings)
    vocabulary = (word_vocab, entity_vocab, sense_vocab, id2wiki_vocab)
    stop_words = loadStopWords(FLAGS.stop_word_file) if FLAGS.stop_word_file is not None else {}

    feature_manager = get_feature_manager(initial_embeddings, FLAGS.embedding_dim,
                 str_sim=FLAGS.str_sim, prior=FLAGS.prior, hasAtt=FLAGS.att,
                 local_context_window=FLAGS.local_context_window,
                  global_context_window=FLAGS.global_context_window)

    # Trim dataset, convert token sequences to integer sequences, crop, and
    # pad. construct data iterator
    logger.Log("Preprocessing data.")
    eval_sets = []
    for i, raw_eval_data in enumerate(raw_eval_sets):
        logger.Log("Processing {} raw eval data ...".format(i))
        AddCandidatesToDocs(raw_eval_sets[i], candidate_handler, topn=FLAGS.topn_candidate,
                            vocab=entity_vocab, logger=logger,
                            include_unresolved=FLAGS.include_unresolved)
        eval_data = PreprocessDataset(raw_eval_sets[i],
                                      vocabulary,
                                      initial_embeddings,
                                      FLAGS.max_tokens,
                                      FLAGS.max_candidates_per_document,
                                      feature_manager,
                                      stop_words=stop_words,
                                      logger=logger,
                                      include_unresolved=FLAGS.include_unresolved,
                                      allow_cropping=FLAGS.allow_cropping)
        eval_sets.append(eval_data)
    training_data_iter = None
    training_data_length = 0
    if raw_training_data is not None:
        logger.Log("Processing raw training data ...")
        AddCandidatesToDocs(raw_training_data, candidate_handler, topn=FLAGS.topn_candidate,
                            vocab=entity_vocab, logger=logger,
                            include_unresolved=FLAGS.include_unresolved)
        training_data = PreprocessDataset(raw_training_data,
                                          vocabulary,
                                          initial_embeddings,
                                          FLAGS.max_tokens,
                                          FLAGS.max_candidates_per_document,
                                          feature_manager,
                                          stop_words=stop_words,
                                          logger=logger,
                                          include_unresolved=FLAGS.include_unresolved,
                                          allow_cropping=FLAGS.allow_cropping)
        training_data_length = training_data.shape[0]
        training_data_iter = MakeTrainingIterator(training_data, FLAGS.batch_size, FLAGS.smart_batching)
    logger.Log("Processing raw eval data ...")
    eval_iterators = []
    for eval_data in eval_sets:
        eval_it = MakeEvalIterator(
            eval_data,
            FLAGS.batch_size)
        eval_iterators.append(eval_it)
    return vocabulary, initial_embeddings, training_data_iter, eval_iterators, training_data_length, feature_manager.base_feature_dim


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
        500,
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

    # Data settings.
    gflags.DEFINE_string("training_data", None,
                         "'type:genre:text_path:mention_file'. text_path or mention file may empty."
                         " use ',' to separate multiple training data.")
    gflags.DEFINE_string("eval_data", None,
                         "'type:genre:text_path:mention_file', text_path or mention file may empty."
                         " use ',' to separate multiple eval data.")

    gflags.DEFINE_string(
        "candidates_file", None, "Each line contains mention-entities pair, separated by tab. "
                                 "type:file, type in ['ppr','wiki_title', 'wiki_anchor', 'wiki_redirect', 'dictionary','yago','ncel']"
                                 "use ',' to separate multiple eval data.")
    gflags.DEFINE_boolean("support_fuzzy", True, "")
    gflags.DEFINE_string(
        "save_candidates_path", None, "Each line: <string><tab><cprob><tab><id>"
                                  "type is ncel.")
    gflags.DEFINE_string("wiki_entity_vocab", None, "line: entity_label \t entity_id")
    gflags.DEFINE_string("wiki_redirect_vocab", None, "line: redirect_id \t entity_id")
    gflags.DEFINE_string("word_embedding_file", None, "")
    gflags.DEFINE_string("entity_embedding_file", None, "")
    gflags.DEFINE_string("sense_embedding_file", None, "")
    gflags.DEFINE_string("stop_word_file", None, "")

    gflags.DEFINE_boolean(
        "allow_cropping",
        False,
        "Trim overly long training examples to fit. If not set, skip them.")
    gflags.DEFINE_integer("max_tokens", 500, "")
    gflags.DEFINE_integer("max_candidates_per_document", 200, "")
    gflags.DEFINE_integer("topn_candidate", 30, "Use all candidates if set 0.")
    gflags.DEFINE_boolean("use_lr_context", True, "Use left and right context.")
    gflags.DEFINE_boolean("split_by_sent", True, "extract context by sent.")

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
    gflags.DEFINE_float("xling", 0.1, "balance candidate loss and mention loss.")
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
    gflags.DEFINE_integer("res_gc_layer_num", 2, "Number of res Graph convolutional layers.")
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
        1000,
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
        FLAGS.experiment_name = "{}-{}".format(
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
        base_feature_dim,
        initial_embeddings,
        logger,
        logfile_header=None):
    # Choose model.
    logger.Log("Building model.")
    if FLAGS.model_type == "NCEL":
        build_model = ncel.build_model
    elif FLAGS.model_type == "MLP":
        build_model = mlp.build_model
    else:
        raise NotImplementedError

    model = build_model(base_feature_dim, initial_embeddings, FLAGS)

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