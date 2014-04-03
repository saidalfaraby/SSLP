from __future__ import division
from collections import defaultdict
import numpy as np


class p_sentence(object):

    def __init__(self, sentence):
        self.words_e = sentence[0]
        self.words_f = sentence[1]


class IBM1(object):

    def __init__(self, p_sentences, converge_thres):
        self.p_sentences = p_sentences
        self.converge_thres = converge_thres
        self.probabilities = None
        self._generate_voc()

    def _generate_voc(self):
        self.voc_e = set()
        self.voc_f = set()

        for sent in p_sentences:
            self.voc_e.update(sent.words_e)
            self.voc_f.update(sent.words_f)

        #self.voc_f.add(None)

    def train(self):
        t = defaultdict(lambda: 1.0/len(self.voc_f))
        converged = False
        iteration = 0
        perplexity_old = 10**200

        while not converged:
            print 'EM iteration %i' % iteration
            # init count(e|f) and total(f)
            count = defaultdict(float)
            total = defaultdict(float)

            # for every pair of sentences in the parallel corpus
            # gather counts
            for sent in self.p_sentences:
                total_s = {}
                for e in sent.words_e:
                    total_s[e] = 0
                    for f in sent.words_f:
                        total_s[e] += t[e, f]
                for e in sent.words_e:
                    for f in sent.words_f:
                        count[e, f] += t[e, f]/total_s[e]
                        total[f] += t[e, f]/total_s[e]

            # normalize and get new t(e|f)
            for f in self.voc_f:
                for e in self.voc_e:
                    t[e, f] = count[e, f] / total[f]

            # have we converged?
            perplexity = 0
            for sent in self.p_sentences:
                mult = 1
                norm = 1/((len(sent.words_f) + 1) ** len(sent.words_e))
                for e in sent.words_e:
                    p_ = 0
                    for f in sent.words_f:
                        p_ += t[e, f]
                    mult *= p_
                perplexity += np.log2(norm * mult)
            perplexity = - perplexity

            if perplexity_old - perplexity < self.converge_thres:
                converged = True
            else:
                perplexity_old = perplexity

            iteration += 1
            for key, value in t.iteritems():
                print key, value


if __name__ == '__main__':
    p_corp = [(['blue', 'house'], ['maison', 'bleu']), (['house'], ['maison'])]

    p_sentences = []
    for sentence in p_corp:
        p_sentences.append(p_sentence(sentence))

    #for p_sent in p_sentences:
    #    print p_sent.words_e, p_sent.words_f

    ibm1 = IBM1(p_sentences, 1e-3)
    ibm1.train()
