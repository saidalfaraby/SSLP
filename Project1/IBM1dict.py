from __future__ import division
from collections import defaultdict
import numpy as np
import dill as pickle
import sys


class Pair_sent(object):

    def __init__(self, sentence):
        self.words_e = sentence[0]
        self.words_f = sentence[1]


class IBM1(object):

    def __init__(self, p_sentences, converge_thres, num_iter=None):
        self.p_sentences = p_sentences
        self.converge_thres = converge_thres
        self.probabilities = None
        self._generate_voc()
        self.num_iter = num_iter

    def _generate_voc(self):
        self.voc_e = set()
        self.voc_f = set()

        for sent in self.p_sentences:
            self.voc_e.update(sent.words_e)
            self.voc_f.update(sent.words_f)

        self.voc_f.add(None)

    def train(self):
        t = defaultdict(lambda: 1.0/len(self.voc_f))
        converged = False
        iteration = 0
        perplexity_old = 10**200

        while not converged:
            print 'EM iteration %i' % (iteration + 1)
            # init count(e|f) and total(f)
            count = defaultdict(float)
            total = defaultdict(float)
            # for every pair of sentences in the parallel corpus
            # gather counts
            # E - Step
            print 'Doing E-Step...'
            for sent in self.p_sentences:
                total_s = {}
                for e in sent.words_e:
                    total_s[e] = 0
                    for f in sent.words_f+[None]:
                        total_s[e] += t[e, f]
                for e in sent.words_e:
                    for f in sent.words_f+[None]:
                        count[e, f] += t[e, f]/total_s[e]
                        total[f] += t[e, f]/total_s[e]

            # normalize and get new t(e|f)
            # M - Step
            print 'Doing M-Step...'
            for e, f in t:
                    t[e, f] = count[e, f] / total[f]

            # have we converged?
            perplexity = 0
            for sent in self.p_sentences:
                mult = 1
                norm = 1/((len(sent.words_f) + 1) ** len(sent.words_e))
                for e in sent.words_e:
                    p_ = 0
                    for f in sent.words_f+[None]:
                        p_ += t[e, f]
                    mult *= p_
                perplexity += np.log2(norm * mult)
            perplexity = - perplexity
            print 'Perplexity: %f' % perplexity

            if perplexity_old - perplexity < self.converge_thres:
                converged = True
                self.probabilities = t
            else:
                perplexity_old = perplexity

            if self.num_iter is not None:
                if iteration + 1 == self.num_iter:
                    self.probabilities = t
                    break

            iteration += 1
            print
            # for key, value in t.iteritems():
            #     print key, value


if __name__ == '__main__':
    #p_corp = [(['blue', 'house'], ['maison', 'bleu']), (['house'], ['maison'])]
    #p_corp = [(['the', 'house'], ['das', 'haus']), (['the', 'book'], ['das', 'buch']), (['a', 'book'], ['ein', 'buch'])]

    n_p_sent = 'all'
    direction = 'fe'
    if len(sys.argv) > 1:
        n_p_sent = int(sys.argv[1])
        if len(sys.argv) > 2:
            direction = sys.argv[2]

    print 'Training in direction: ' + direction

    p_corp = []
    print 'Training for ' + str(n_p_sent) + ' sentences...'
    with open('corpus.en', 'rb') as corpus_en:
        with open('corpus.nl', 'rb') as corpus_nl:
            for line_en, line_nl in zip(corpus_en.readlines(), corpus_nl.readlines()):
                if direction == 'ef':
                    p_corp.append((line_en.split(), line_nl.split()))
                elif direction == 'fe':
                    p_corp.append((line_nl.split(), line_en.split()))
                if n_p_sent is not 'all':
                    if len(p_corp) == n_p_sent:
                        break

    p_sentences = []
    for sentence in p_corp:
        p_sentences.append(Pair_sent(sentence))

    ibm1 = IBM1(p_sentences, 1e-1, num_iter=20)
    ibm1.train()

    print 'Saving the model to disk...'
    if direction == 'ef':
        with open('IBM1_trained_ef.pickle', 'wb') as handle:
            pickle.dump(ibm1, handle)
    elif direction == 'fe':
        with open('IBM1_trained_fe.pickle', 'wb') as handle:
            pickle.dump(ibm1, handle)

    if direction == 'ef':
        key = ('this', 'deze')
        print key, ibm1.probabilities[key[0], key[1]]
        key = ('these', 'deze')
        print key, ibm1.probabilities[key[0], key[1]]
        key = ('transparency', 'transparantie')
        print key, ibm1.probabilities[key[0], key[1]]
    elif direction == 'fe':
        key = ('deze', 'this')
        print key, ibm1.probabilities[key[0], key[1]]
        key = ('deze', 'these')
        print key, ibm1.probabilities[key[0], key[1]]
        key = ('transparantie', 'transparency')
        print key, ibm1.probabilities[key[0], key[1]]
