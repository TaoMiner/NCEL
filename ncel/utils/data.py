# -*- coding: utf-8 -*-
"""Dataset handling and related yuck."""

import random
import time
import sys
import struct

import numpy as np
from ncel.utils.layers import buildGraph
from ncel.utils.Candidates import resortCandidates

from ncel.utils.misc import Accumulator

PADDING_TOKEN = "_PAD"
# UNK must be existed in pre-trained embeddings
UNK_TOKEN = "_"

CORE_VOCABULARY = {PADDING_TOKEN: 0}

PADDING_ID = CORE_VOCABULARY[PADDING_TOKEN]

class SimpleProgressBar(object):
    """ Simple Progress Bar and Timing Snippet
    """

    def __init__(self, msg=">", bar_length=80, enabled=True):
        super(SimpleProgressBar, self).__init__()
        self.enabled = enabled
        if not self.enabled:
            return

        self.begin = time.time()
        self.bar_length = bar_length
        self.msg = msg

    def step(self, i, total):
        if not self.enabled:
            return
        sys.stdout.write('\r')
        pct = (i / float(total)) * 100
        ii = i * self.bar_length // total
        fmt = "%s [%-{}s] %d%% %ds / %ds    ".format(self.bar_length)
        total_time = time.time() - self.begin
        expected = total_time / ((i + 1e-03) / float(total))
        sys.stdout.write(fmt % (self.msg, '=' * ii, pct, total_time, expected))
        sys.stdout.flush()

    def reset(self):
        if not self.enabled:
            return
        self.begin = time.time()

    def finish(self):
        if not self.enabled:
            return
        self.reset()
        sys.stdout.write('\n')

def recordCandidates(A, rank, length):
    # avg rank
    if rank >= 0 and rank < 30:
        rank_count = len(A.get('rank_30', clear=False))
        if rank_count >= A.maxlen:
            A.add('avg_rank_30', A.get_avg('rank_30'))
        A.add('rank_30', rank/float(length))
    elif rank >= 30 and rank < 50:
        rank_count = len(A.get('rank_50', clear=False))
        if rank_count >= A.maxlen:
            A.add('avg_rank_50', A.get_avg('rank_50'))
        A.add('rank_50', rank / float(length))
    else:
        rank_count = len(A.get('rank_large', clear=False))
        if rank_count >= A.maxlen:
            A.add('avg_rank_large', A.get_avg('rank_large'))
        A.add('rank_large', rank / float(length))

def statsCandidates(A, logger=None):
    rank_count_30_avg = len(A.get('avg_rank_30', clear=False)) * A.maxlen
    rank_count_30 = len(A.get('rank_30', clear=False))
    rank_count_50_avg = len(A.get('avg_rank_50', clear=False)) * A.maxlen
    rank_count_50 = len(A.get('rank_50', clear=False))
    rank_count_large_avg = len(A.get('avg_rank_large', clear=False)) * A.maxlen
    rank_count_large = len(A.get('rank_large', clear=False))

    if rank_count_30_avg > 0:
        avg_rank_30 = (rank_count_30_avg * A.get_avg('avg_rank_30') + rank_count_30 * A.get_avg('rank_30')) / (rank_count_30_avg+rank_count_30)
    else: avg_rank_30 = A.get_avg('rank_30')
    if rank_count_50_avg > 0:
        avg_rank_50 = (rank_count_50_avg * A.get_avg('avg_rank_50') + rank_count_50 * A.get_avg('rank_50')) / (rank_count_50_avg + rank_count_50)
    else: avg_rank_50 = A.get_avg('rank_50')
    if rank_count_large_avg > 0:
        avg_rank_large = (rank_count_large_avg * A.get_avg('avg_rank_large') + rank_count_large * A.get_avg('rank_large')) / (rank_count_large_avg + rank_count_large)
    else: avg_rank_large = A.get_avg('rank_large')
    if logger is not None:
        logger.Log("avg gold rank 30:{}/{}, 50:{}/{}, large:{}/{}.".format(avg_rank_30, rank_count_30_avg+rank_count_30,
                       avg_rank_50,rank_count_50_avg+rank_count_50, avg_rank_large, rank_count_large+rank_count_large_avg))

def AddCandidatesToDocs(dataset, candidate_handler, vocab=None, topn=0,
                        include_unresolved=False, logger=None):
    for i, doc in enumerate(dataset):
        candidate_handler.add_candidates_to_document(dataset[i], vocab=vocab,topn=topn)
    if logger is not None:
        logger.Log("Add candidates success: totally {} candidates of {} mentions in {} documents!".format(
            sum([doc.n_candidates for doc in dataset]), sum([len(doc.mentions) for doc in dataset]), len(dataset)))

def BuildVocabulary(raw_training_data, raw_eval_sets, word_embedding_path, logger=None):
    # Find the set of words that occur in the data.
    logger.Log("Constructing vocabulary...")

    words_in_data = set()
    mentions_in_data = set()
    datasets = []
    if raw_training_data is not None:
        datasets.append(raw_training_data)
    for eval_set in raw_eval_sets:
        datasets.append(eval_set)
    for i, dataset in enumerate(datasets):
        for j, doc in enumerate(dataset):
            words_in_data.update(doc.tokens)
            for k, mention in enumerate(doc.mentions):
                if not mention._is_trainable : continue
                mentions_in_data.add(mention._mention_str)
                datasets[i][j].mentions[k].updateSentIdxByTokenIdx()
        # update mention sent index

    logger.Log("Found " + str(len(words_in_data)) + " word types.")
    logger.Log("Found " + str(len(mentions_in_data)) + " mention types.")

    # Build a vocabulary of words in the data for which we have an
    # embedding.
    word_vocabulary = BuildVocabularyForBinaryEmbeddingFile(
        word_embedding_path, words_in_data, CORE_VOCABULARY)

    return word_vocabulary, mentions_in_data

def BuildEntityVocabulary(candidate_entities, entity_embedding_file, sense_embedding_file=None, logger=None):
    # Find the set of words that occur in the data.
    logger.Log("Constructing entity vocabulary...")

    logger.Log("Found " + str(len(candidate_entities)) + " entity types.")

    entity_vocabulary = BuildVocabularyForBinaryEmbeddingFile(
        entity_embedding_file, candidate_entities, CORE_VOCABULARY)

    # Build a vocabulary of entities in the data for which we have an
    # embedding.
    sense_vocabulary = BuildVocabularyForBinaryEmbeddingFile(sense_embedding_file,
                        candidate_entities, CORE_VOCABULARY, isSense=True) if sense_embedding_file is not None else None

    return entity_vocabulary, sense_vocabulary

def BuildVocabularyForBinaryEmbeddingFile(path, types_in_data, core_vocabulary, isSense=False):
    """Quickly iterates through a GloVe-formatted text vector file to
    extract a working vocabulary of words that occur both in the data and
    in the vector file."""

    num_embeddings = 4 if not isSense else 8

    vocabulary = {}
    vocabulary.update(core_vocabulary)
    next_index = len(vocabulary)
    vocab_size = 0
    with open(path, 'rb') as f:
        # read file head: vocab size and layer size
        char_set = []
        while True:
            ch = f.read(1)
            if ch == b' ' or ch == b'\t':
                vocab_size = (int)(b''.join(char_set).decode())
                del char_set[:]
                continue
            if ch == b'\n':
                layer_size = (int)(b''.join(char_set).decode())
                break
            char_set.append(ch)
        for i in range(vocab_size):
            # read entity label
            del char_set[:]
            while True:
                ch = struct.unpack('c', f.read(1))[0]
                # add split interval white space
                if ch == b' ' or ch == b'\t':
                    break
                char_set.append(ch)
            word = b''.join(char_set)
            word = word.decode('utf-8', 'ignore')
            f.read(num_embeddings * layer_size)
            if word in types_in_data and word not in vocabulary:
                vocabulary[word] = next_index
                next_index += 1
            f.read(1)  # \n
    return vocabulary

def initVectorFormat(size):
    tmp_struct_fmt = []
    for i in range(size):
        tmp_struct_fmt.append('f')
    p_struct_fmt = "".join(tmp_struct_fmt)
    return p_struct_fmt

def LoadEmbeddingsFromBinary(vocabulary, embedding_dim, path, isSense=False):
    """Prepopulates a numpy embedding matrix indexed by vocabulary with
    values from a vector file.

    For now, values not found in the file will be set to zero."""
    loaded = 0
    emb = np.zeros( (len(vocabulary), embedding_dim), dtype=np.float32)
    emb_mu = np.zeros((len(vocabulary), embedding_dim), dtype=np.float32) if isSense else None
    vocab_size = 0
    layer_size = 0
    with open(path, 'rb') as f:
        # read file head: vocab size and layer size
        char_set = []
        while True:
            ch = f.read(1)
            if ch == b' ' or ch == b'\t':
                vocab_size = (int)(b''.join(char_set).decode())
                del char_set[:]
                continue
            if ch == b'\n':
                layer_size = (int)(b''.join(char_set).decode())
                break
            char_set.append(ch)
        assert layer_size == embedding_dim, "No matched embeddings dimension."
        p_struct_fmt = initVectorFormat(embedding_dim)
        for i in range(vocab_size):
            # read entity label
            del char_set[:]
            while True:
                ch = struct.unpack('c', f.read(1))[0]
                # add split interval white space
                if ch == b' ' or ch == b'\t':
                    break
                char_set.append(ch)
            word = b''.join(char_set)
            word = word.decode('utf-8', 'ignore')
            tmp_vec = np.array(struct.unpack(p_struct_fmt, f.read(4 * embedding_dim)), dtype=float)
            # context cluster senter
            if isSense:
                tmp_mu = np.array(struct.unpack(p_struct_fmt, f.read(4 * embedding_dim)), dtype=float)
            if word in vocabulary:
                emb[vocabulary[word], :] = tmp_vec
                if isSense:
                    emb_mu[vocabulary[word], :] = tmp_mu
                loaded += 1
            f.read(1)  # \n
    assert loaded > 0, "No word embeddings of correct size found in file."
    if isSense:
        return emb, emb_mu
    else:
        return emb

# preprocess raw data
# todo: may be not to trim dataset
def TrimDataset(dataset, max_tokens, allow_cropping=True, logger=None):
    """Avoid using excessively long training examples."""
    # disgard over length sentence and document
    if max_tokens > 0:
        diff_text = sum([1 for doc in dataset if len(doc.tokens) <= max_tokens])
        if not allow_cropping:
            if logger and diff_text > 0:
                logger.Log(
                    "Discarding " +
                    str(diff_text) +
                    " textual over-length documents.")
            dataset = [doc for doc in dataset if len(doc.tokens) <= max_tokens]
        else:
            if logger and diff_text > 0:
                logger.Log(
                    "Cropping " +
                    str(diff_text) +
                    " textual over-length documents.")
            # cropping
            for i, doc in enumerate(dataset):
                if len(doc.tokens) > max_tokens:
                    # trim doc
                    dataset[i].tokens = dataset[i].tokens[:max_tokens]
                    new_sents = []
                    sent_len = 0
                    sent_idx = 0
                    while sent_len < max_tokens and sent_idx < len(doc.sentences):
                        diff = max_tokens - sent_len - len(doc.sentences[sent_idx])
                        if diff >= 0 :
                            new_sents.append(doc.sentences[sent_idx])
                            sent_len += len(doc.sentences[sent_idx])
                        else:
                            new_sents.append(doc.sentences[sent_idx][:diff])
                            sent_len += len(doc.sentences[sent_idx][:diff])
                        sent_idx += 1
                    dataset[i].sentences = new_sents
                    # trim mentions
                    for j, mention in enumerate(doc.mentions):
                        if mention._mention_end > max_tokens:
                            dataset[i].mentions[j]._is_trainable = False
    return dataset

def TokensToIDs(word_vocabulary, dataset, stop_words=None, logger=None):
    """Replace strings in original boolean dataset with token IDs."""

    tokens = 0
    unks = 0
    lowers = 0
    raises = 0

    if UNK_TOKEN in CORE_VOCABULARY:
        unk_id = CORE_VOCABULARY[UNK_TOKEN]
    else: unk_id = -1

    for i, doc in enumerate(dataset):
        # tokens to id
        for j, sent in enumerate(doc.sentences):
            for k, token in enumerate(sent):
                if stop_words is not None and token in stop_words:
                    dataset[i].sentences[j][k] = unk_id
                    unks += 1
                elif token in word_vocabulary:
                    dataset[i].sentences[j][k] = word_vocabulary[token]
                elif token.lower() in word_vocabulary:
                    dataset[i].sentences[j][k] = word_vocabulary[token.lower()]
                    lowers += 1
                elif token.upper() in word_vocabulary:
                    dataset[i].sentences[j][k] = word_vocabulary[token.upper()]
                    raises += 1
                else:
                    dataset[i].sentences[j][k] = unk_id
                    unks += 1
                tokens += 1
    if logger:
        logger.Log("Unk rate {:2.6f}%, downcase rate {:2.6f}%, upcase rate {:2.6f}%".format(
            (unks * 100.0 / tokens), (lowers * 100.0 / tokens), (raises * 100.0 / tokens)))
    # filter unk tokens if not assign id in vocab
    if unk_id == -1:
        for i, doc in enumerate(dataset):
            # new sents filter out unk at token level, keep empty sents
            empty_sent_count = 0
            empty_cumsum = []
            new_sents = [[token for token in sent if token != unk_id] for sent in doc.sentences]
            for sent in new_sents:
                empty_cumsum.append(empty_sent_count)
                if len(sent) == 0: empty_sent_count += 1

            for j, mention in enumerate(doc.mentions):
                if not mention._is_trainable: continue
                sent = doc.sentences[mention._sent_idx]
                if len(new_sents[mention._sent_idx]) == 0:
                    dataset[i].mentions[j]._is_trainable = False
                    continue
                mt_unks = 0
                for t in sent[mention._pos_in_sent:mention._pos_in_sent + mention._mention_length]:
                    if t == unk_id: mt_unks += 1
                if mt_unks > 0: dataset[i].mentions[j]._mention_length -= mt_unks
                # update mention sent index by new sents
                empty_token_in_sent = 0
                for t in sent[:mention._pos_in_sent]:
                    if t == unk_id: empty_token_in_sent += 1
                dataset[i].mentions[j]._sent_idx -= empty_cumsum[mention._sent_idx]
                dataset[i].mentions[j]._pos_in_sent -= empty_token_in_sent
            # update doc sentences by removing both unk token and empty sents
            dataset[i].sentences = [sent for sent in new_sents if len(sent) > 0]
            # update mention offset of tokens
            for j, mention in enumerate(doc.mentions):
                dataset[i].mentions[j].updateTokenIdxBySentIdx()
            # update doc tokens
            doc.tokens = [t for s in dataset[i].sentences for t in s]
    return dataset

def EntityToIDs(entity_vocabulary, dataset, sense_vocab=None,
                include_unresolved=False, logger=None):

    if UNK_TOKEN in CORE_VOCABULARY:
        unk_id = CORE_VOCABULARY[UNK_TOKEN]
    else: unk_id = -1
    # avg rank
    # A = Accumulator(maxlen=500)

    m_num = 0
    m_unk = 0
    c_num = 0
    c_unk = 0
    nil_num = 0
    g_unk = 0
    no_gold_cand = 0

    for i, doc in enumerate(dataset):
        dataset[i].n_candidates = 0
        for j, mention in enumerate(doc.mentions):
            m_num += 1
            if not mention._is_trainable :
                m_unk += 1
                continue
            if include_unresolved and mention._is_NIL:
                dataset[i].mentions[j]._gold_ent_id = unk_id
                nil_num += 1
            elif not include_unresolved and mention._is_NIL:
                mention._is_trainable = False
                m_unk += 1
            else:
                # set candidate label
                is_trainable = False
                for k, cand in enumerate(mention.candidates):
                    if mention.gold_ent_id() is not None and cand.id == mention.gold_ent_id():
                        dataset[i].mentions[j].candidates[k].setGold()
                        is_trainable = True
                        break
                if not is_trainable:
                    mention._is_trainable = False
                    m_unk += 1
                    no_gold_cand += 1
                # gold no vec
                elif mention.gold_ent_id() is None or mention.gold_ent_id() not in entity_vocabulary:
                    mention._is_trainable = False
                    m_unk += 1
                    g_unk += 1
                else:
                    if sense_vocab is not None and mention.gold_ent_id() in sense_vocab:
                        dataset[i].mentions[j]._gold_sense_id = sense_vocab[mention.gold_ent_id()]
                    else : dataset[i].mentions[j]._gold_sense_id = unk_id
                    dataset[i].mentions[j]._gold_ent_id = entity_vocabulary[mention.gold_ent_id()]
            if mention._is_trainable:
                # replace candidate id
                new_candidates = []
                # avg rank
                # rank = -1
                for k, cand in enumerate(mention.candidates):
                    c_num += 1
                    if cand.id in entity_vocabulary:
                        if sense_vocab is not None and cand.id in sense_vocab:
                            dataset[i].mentions[j].candidates[k].setSense(sense_vocab[cand.id])
                        else : dataset[i].mentions[j].candidates[k].setSense(unk_id)
                        dataset[i].mentions[j].candidates[k].id = entity_vocabulary[cand.id]
                        new_candidates.append(dataset[i].mentions[j].candidates[k])
                        # avg rank
                        # if cand.getIsGlod(): rank = k
                    else:
                        c_unk += 1
                        if cand.getIsGlod():
                            no_gold_cand += 1
                            m_unk += 1
                            mention._is_trainable = False
                dataset[i].mentions[j].candidates = new_candidates
                dataset[i].n_candidates += len(new_candidates)
                # if rank>=0:
                #    recordCandidates(A, rank, len(new_candidates))
    if logger:
        # avg rank
        # statsCandidates(A, logger=logger)
        logger.Log("Untrainable rate {:2.6f}% ({}/{}), "
          "Unk candidate rate {:2.6f}% ({}/{}), "
           "No gold candidate rate {:2.6f}% ({}/{}), "
          "Unk gold rate {:2.6f}% ({}/{}), "
          "NIL rate {:2.6f}% ({}/{}) !".format((m_unk*100.0/m_num), m_unk, m_num,
              (c_unk*100.0/c_num), c_unk, c_num,
               (no_gold_cand * 100.0/m_num), no_gold_cand, m_num,
                (g_unk * 100.0 / m_num), g_unk, m_num,
                (nil_num * 100.0 / m_num), nil_num, m_num))
    return dataset

def CropMentionAndCandidates(dataset, max_candidates, allow_cropping=True, logger=None):
    raw_doc_num = len(dataset)
    # over mention-candidate_pairs size that may be cropped
    cropped_dataset = [doc for doc in dataset if doc.n_candidates <= max_candidates]

    diff_doc = raw_doc_num - len(cropped_dataset)

    if not allow_cropping:
        if logger and diff_doc > 0:
            logger.Log(
                "Discarding " +
                str(diff_doc) +
                " candidate over-length documents.")
        dataset = cropped_dataset
    else:
        if logger and diff_doc > 0:
            logger.Log(
                "Cropping " +
                str(diff_doc) +
                " candidate over-length documents.")

        cropped_m = 0
        cropped_d = 0

        for i, doc in enumerate(dataset):
            # crop candidate over length doc by make mention intrainable
            if doc.n_candidates > max_candidates:
                candidate_length_set = [[m_idx, len(m.candidates)] for m_idx, m in enumerate(doc.mentions)]
                cropped_d += 1
                c2s_ms = sorted(candidate_length_set, key=lambda x: x[1], reverse=True)
                diff = doc.n_candidates - max_candidates
                p = -1
                while diff > 0 and p < len(c2s_ms)-1:
                    p += 1
                    if not dataset[i].mentions[c2s_ms[p][0]]._is_trainable : continue
                    dataset[i].mentions[c2s_ms[p][0]]._is_trainable = False
                    cropped_m += 1
                    tmp_clen = len(doc.mentions[c2s_ms[p][0]].candidates)
                    diff -= tmp_clen
                    dataset[i].n_candidates -= tmp_clen
                if p == len(c2s_ms)-1 : dataset[i].n_candidates = 0

        logger.Log("Actual cropped {} mentions of {} documents! ".format(cropped_m, cropped_d))

    dataset = [doc for doc in dataset if doc.n_candidates > 0]
    for i, doc in enumerate(dataset):
        # filter out cannot trainable mentions
        dataset[i].mentions = [mention for mention in doc.mentions if mention._is_trainable]

    logger.Log("Remove {} docs!".format(raw_doc_num - len(dataset)))

    return dataset

# adj : node * node
def PadDocument(
        x, adj, y,
        length,
        allow_cropping=True):
    paddings = length - x.shape[0]
    feature_dim = x.shape[1]
    if paddings < 0:
        if not allow_cropping:
            raise NotImplementedError(
                "Cropping not allowed. "
                "Please set max_candidates_per_document to some sufficiently large value or allow_cropping.")
        x = x[:length, :]
        y = y[:length]
        adj = adj[:length, :length]
    elif paddings>0:
        x_pad = np.zeros((paddings, feature_dim), dtype=float)
        pad = np.zeros(paddings, dtype=float)
        x = np.concatenate((x, x_pad), axis=0)
        y = np.concatenate((y, pad), axis=0)
        tmp_adj = np.zeros((length, length))
        node_num = adj.shape[0]
        for i in range(node_num):
            for j in range(node_num):
                tmp_adj[i][j] = adj[i][j]
        adj = tmp_adj
    return x, adj, y

# process raw data
def PreprocessDataset(
        dataset,
        vocabulary,
        embeddings,
        max_tokens,
        max_candidates,
        feature_manager,
        stop_words={},
        logger=None,
        include_unresolved=False,
        allow_cropping=False):

    _, entity_embeddings, _, _ = embeddings
    word_vocab, entity_vocab, sense_vocab, _ = vocabulary
    dataset = TokensToIDs(word_vocab, dataset, stop_words=stop_words, logger=logger)

    dataset = TrimDataset(dataset, max_tokens, allow_cropping=allow_cropping, logger=logger)

    dataset = EntityToIDs(entity_vocab, dataset, sense_vocab=sense_vocab,
                          include_unresolved=include_unresolved, logger=logger)
    dataset = CropMentionAndCandidates(dataset, max_candidates, logger=logger)
    # inspectDoc(dataset[0], word_vocab=word_vocabulary)
    for i, doc in enumerate(dataset):
        feature_manager.setBaseFeature(dataset[i])
    if logger is not None:
        logger.Log("After crop and filter: totally {} candidates of {} mentions in {} documents!".format(
            sum([len(doc.n_candidates) for doc in dataset]), sum([len(doc.mentions) for doc in dataset]), len(dataset)))
    return np.array(dataset)

def MakeTrainingIterator(
        sources,
        batch_size,
        smart_batches=True):

    def build_batches():
        dataset_size = len(sources)
        order = list(range(dataset_size))
        random.shuffle(order)
        order = np.array(order)

        num_splits = 10  # TODO: Should we be smarter about split size?
        order_limit = len(order) // num_splits * num_splits
        order = order[:order_limit]
        order_splits = np.split(order, num_splits)
        batches = []

        for split in order_splits:
            # Put indices into buckets based on candidate size.
            keys = []
            for i in split:
                n_candidates = sources[i].n_candidates
                keys.append((i, n_candidates))
            keys = sorted(keys, key=lambda __key: __key[1])

            # Group indices from buckets into batches, so that
            # examples in each batch have similar length.
            batch = []
            for i, _ in keys:
                batch.append(i)
                if len(batch) == batch_size:
                    batches.append(batch)
                    batch = []
        return batches

    def batch_iter():
        batches = build_batches()
        num_batches = len(batches)
        idx = -1
        order = list(range(num_batches))
        random.shuffle(order)

        while True:
            idx += 1
            if idx >= num_batches:
                # Start another epoch.
                batches = build_batches()
                num_batches = len(batches)
                idx = 0
                order = list(range(num_batches))
                random.shuffle(order)
            batch_indices = batches[order[idx]]
            yield sources[batch_indices]

    def data_iter():
        dataset_size = len(sources)
        start = -1 * batch_size
        order = list(range(dataset_size))
        random.shuffle(order)

        while True:
            start += batch_size
            if start > dataset_size - batch_size:
                # Start another epoch.
                start = 0
                random.shuffle(order)
            batch_indices = order[start:start + batch_size]
            yield sources[batch_indices]

    train_iter = batch_iter if smart_batches else data_iter

    return train_iter()


def MakeEvalIterator(
        sources,
        batch_size):
    # Make a list of minibatches from a dataset to use as an iterator.
    # TODO(SB): Pad out the last few examples in the eval set if they don't
    # form a batch.

    print("Warning: May be discarding eval examples at batch ends.")

    dataset_size = len(sources)
    order = list(range(dataset_size))
    data_iter = []
    start = -batch_size
    while True:
        start += batch_size

        if start >= dataset_size:
            break

        batch_indices = order[start:start + batch_size]
        candidate_batch = sources[batch_indices]

        if len(candidate_batch) == batch_size:
            data_iter.append(candidate_batch)
        else:
            print("Skipping " + str(len(candidate_batch)) + " examples.")
    return data_iter