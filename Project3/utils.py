from __future__ import division
from collections import defaultdict
import numpy as np
import nltk
from FeatureExtraction import Features


def kl_divergence(lm1, lm2):
    diff = []
    for w in lm1:
        try:
            val = np.log(lm1[w]/lm2[w])
            if np.isnan(val) or np.isinf(val):
                val = 0.0
            else:
                val *= lm1[w]
        except ZeroDivisionError:
            val = 0.0

        diff.append(val)

    return sum(diff)


def feat_sent(uni_sentence, bi_sentence, uni_lm, bi_lm):
    features = []
    tot_p_uni = 1
    for w in uni_sentence:
        tot_p_uni *= uni_lm[w]
    features.append(tot_p_uni)

    tot_p_bi = 1
    for bi in bi_sentence:
        tot_p_bi *= bi_lm[bi]
    features.append(tot_p_bi)

    return np.asarray(features)


def create_dataset(uni_sentences, bi_sentences, uni_lm, bi_lm):
    data = []
    for i, sent in enumerate(uni_sentences):
        data.append(feat_sent(sent, bi_sentences[i], uni_lm, bi_lm))

    return np.asarray(data)


def create_w2v_dataset(sentences, model):
    data = []
    for sent in sentences:
        feat_vec = []
        for w in sent:
            feat_vec.append(model[w])
        data.append(sum(feat_vec)/len(sent))

    return np.asarray(data)


def create_tfidf_data(sentences):
    X_train = np.array([''.join(el) for el in sentences])
    print X_train
