from __future__ import division
from collections import defaultdict
import nltk
import numpy as np
from sklearn import svm
from gensim.models import Word2Vec

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# we will take log(0) = -Inf so turn off this warning
np.seterr(divide='ignore')


class LanguageModel:

    def __init__(self, domain):
        self.domain = domain
        self.uni_probs = defaultdict(float)
        self.bi_probs = defaultdict(float)

    def build_simple_lm(self, uni_sentences, bi_sentences):
        total_occ_uni = 0
        total_occ_bi = 0
        for s in uni_sentences:
            total_occ_uni += len(s)
            for t in s:
                self.uni_probs[t] += 1.0

        for s in bi_sentences:
            total_occ_bi += len(s)
            for t in s:
                self.bi_probs[t] += 1.0

        for t in self.uni_probs:
            self.uni_probs[t] /= total_occ_uni

        for t in self.bi_probs:
            self.bi_probs[t] /= total_occ_bi

        # simple smoothing
        self.uni_probs.default_factory = lambda: 1.0/(total_occ_uni + 0.3 * total_occ_uni)
        self.bi_probs.default_factory = lambda: 1.0/(total_occ_bi + 0.3 * total_occ_bi)


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
        data.append(sum(feat_vec))

    return np.asarray(data)


def main():
    domain = 'legal.half.en'
    with open('project3_data_selection/'+domain, 'rb') as doc:
        uni_sentences = [sentence.split() for sentence in doc.readlines()]

    domain_out = 'out.mixed.legal.en'
    with open('project3_data_selection/'+domain_out, 'rb') as doc:
        uni_sentences_out = [sentence.split() for sentence in doc.readlines()]

    bi_sentences = [nltk.bigrams(sentence) for sentence in uni_sentences]
    in_lm = LanguageModel(domain)
    in_lm.build_simple_lm(uni_sentences, bi_sentences)

    bi_sentences_out = [nltk.bigrams(s) for s in uni_sentences_out]
    out_lm = LanguageModel(domain_out)
    out_lm.build_simple_lm(uni_sentences_out, bi_sentences_out)

    print 'KL(Out||In) =', kl_divergence(out_lm.uni_probs, in_lm.uni_probs)
    print 'KL(In||Out) =', kl_divergence(in_lm.uni_probs, out_lm.uni_probs)
    print
    print 'KL(Out||In) =', kl_divergence(out_lm.bi_probs, in_lm.bi_probs)
    print 'KL(In||Out) =', kl_divergence(in_lm.bi_probs, out_lm.bi_probs)


def main2():
    domain = 'legal.half.en'
    with open('project3_data_selection/'+domain, 'rb') as doc:
        uni_sentences = [sentence.split() for sentence in doc.readlines()]
    model = Word2Vec(uni_sentences, min_count=1, workers=4, size=10)
    model.save('w2vec_legal')


def main3():
    domain = 'legal.half.en'
    with open('project3_data_selection/'+domain, 'rb') as doc:
        uni_sentences = [sentence.split() for sentence in doc.readlines()]
    model = Word2Vec.load('w2vec_legal')

    print 'Getting training data...'
    train_data = create_w2v_dataset(uni_sentences, model)

    print 'Fitting one class SVM'
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(train_data)

    print 'Getting testing data...'
    domain_out = 'out.mixed.legal.en'
    with open('project3_data_selection/'+domain_out, 'rb') as doc:
        uni_sentences_out = [sentence.split() for sentence in doc.readlines()]

    test_data = create_w2v_dataset(uni_sentences_out, model)

    print 'Predicting for test data...'
    predictions = clf.predict(test_data)
    print predictions
    print np.unique(predictions)

if __name__ == '__main__':
    main3()
