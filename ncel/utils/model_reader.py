# -*- coding: utf-8 -*-
import numba
import numpy as np

from ncel.utils.data import LoadEmbeddingsFromBinary

class ModelReader(object):
    def __init__(self, path, dim=300):
        model_files = path.split(",")
        assert len(model_files) == 4, print("Error yamada model!")
        self._w2v_file = model_files[0]
        self._e2v_file = model_files[1]
        self._w_file = model_files[2]
        self._b_file = model_files[3]
        self._dim = dim

        self._word_embedding = None
        self._entity_embedding = None
        self._W = None
        self._b = None

    def loadModel(self, word_vocab, entity_vocab):
        self._word_embedding = LoadEmbeddingsFromBinary(word_vocab, self._dim,
                                            self._w2v_file, isSense=False)
        self._entity_embedding = LoadEmbeddingsFromBinary(entity_vocab, self._dim,
                                            self._e2v_file, isSense=False)
        self._W = np.load(self._w_file)
        self._b = np.load(self._b_file)

    @property
    def word_embedding(self):
        return self._word_embedding

    @property
    def entity_embedding(self):
        return self._entity_embedding

    @property
    def W(self):
        return self._W

    @property
    def b(self):
        return self._b

    def get_word_vector(self, word, default=None):
        return self._word_embedding[word] if word in self._word_embedding else default

    def get_entity_vector(self, entity, default=None):
        return self._entity_embedding[entity] if entity in self._entity_embedding else default

    def get_text_embed(self, batch_tokens):
        return self.embed(batch_tokens)

    def get_text_array(self, text):
        vectors = [self.get_word_vector(t) for t in text]
        vectors = [v for v in vectors if v is not None]
        return vectors

    @numba.autojit
    def get_text_vector(self, vectors):
        if vectors is None or len(vectors) < 1:
            return None
        ret = np.mean(vectors, axis=0)
        ret = np.dot(ret, self._W)
        ret += self._b

        ret /= np.linalg.norm(ret, 2)

        return ret
