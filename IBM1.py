from __future__ import division
from collections import defaultdict
import numpy as np
import dill as pickle
import shelve


class Pair_sent(object):

    def __init__(self, sentence):
        self.words_e = sentence[0]
        self.words_f = sentence[1]


class IBM1(object):

    def __init__(self, p_sentences, converge_thres):
        self.p_sentences = p_sentences
        self.converge_thres = converge_thres
        self.probabilities = None
        self._generate_voc()
        self._generate_dict_ind()

    def _generate_voc(self):
        self.voc_e = set()
        self.voc_f = set()

        for sent in self.p_sentences:
            self.voc_e.update(sent.words_e)
            self.voc_f.update(sent.words_f)

        self.voc_f.add(None)
        self.voc_e = list(self.voc_e)
        self.voc_f = list(self.voc_f)

    def _generate_dict_ind(self):
        # self.dict_e = shelve.open('e_to_id.shelf')
        # for i in xrange(len(self.voc_e)):
        #     self.dict_e[self.voc_e[i]] = i
        # self.dict_f = shelve.open('f_to_id.shelf')
        # for i in xrange(len(self.voc_f)):
        #     self.dict_f[self.voc_f[i]] = i
        self.dict_e = {key: value for (key, value) in zip(self.voc_e, range(len(self.voc_e)))}
        self.dict_f = {key: value for (key, value) in zip(self.voc_f, range(len(self.voc_f)))}

    def train(self):
        # import shelve
        #t = anydbm.open('prob_table', 'c')
        #t = shelve.open('prob_table.shelf')
        #t = dbdict("prob_table")
        # t = defaultdict(lambda: 1.0/len(self.voc_f))
        # t = {}
        t = np.ones((len(self.voc_e), len(self.voc_f))) * (1.0/len(self.voc_f))
        # i = 0
        # n = len(self.voc_e) * len(self.voc_f)
        # for e in self.voc_e:
        #     for f in self.voc_f:
        #         print str(i)+' of '+str(n)
        #         i += 1
        #         t[str((e, f))] = 1.0/len(self.voc_f)
        print 'Finished creating.'
        #print 'Initial probabilities: %f' % (1.0/len(self.voc_f))
        converged = False
        iteration = 0
        perplexity_old = 10**200
        #count = defaultdict(float)
        count = np.zeros((len(self.voc_e), len(self.voc_f)))
        total = defaultdict(float)

        while not converged:
            print 'EM iteration %i' % iteration
            # init count(e|f) and total(f)
            # count.clear()
            count = np.zeros((len(self.voc_e), len(self.voc_f)))
            total.clear()
            # for every pair of sentences in the parallel corpus
            # gather counts
            # E - Step
            for sent in self.p_sentences:
                # total_s = {}
                total_s = {key: value for (key, value) in zip(sent.words_e, [sum([t[self.dict_e[e], self.dict_f[f]] for f in sent.words_f+[None]]) for e in sent.words_e])}
                # for e in sent.words_e:
                #     total_s[e] = sum([t[e, f] for f in sent.words_f+[None]])
                for e in sent.words_e:
                    for f in sent.words_f+[None]:
                        count[self.dict_e[e], self.dict_f[f]] += t[self.dict_e[e], self.dict_f[f]]/total_s[e]
                        total[f] += t[self.dict_e[e], self.dict_f[f]]/total_s[e]

            # normalize and get new t(e|f)
            # M - Step
            # t = {key: value for (key, value) in zip([(e, f) for f in self.voc_f for e in self.voc_e], [count[e, f]/total[f] for f in self.voc_f for e in self.voc_e])}
            for f in self.voc_f:
                for e in self.voc_e:
                    t[self.dict_e[e], self.dict_f[f]] = count[self.dict_e[e], self.dict_f[f]] / total[f]

            # have we converged?
            perplexity = 0
            for sent in self.p_sentences:
                mult = 1
                norm = 1/((len(sent.words_f) + 1) ** len(sent.words_e))
                for e in sent.words_e:
                    p_ = sum([t[self.dict_e[e], self.dict_f[f]] for f in sent.words_f+[None]])
                    # for f in sent.words_f+[None]:
                    #     p_ += t[e, f]
                    mult *= p_
                perplexity += np.log2(norm * mult)
            perplexity = - perplexity
            print 'Perplexity: %f' % perplexity

            if perplexity_old - perplexity < self.converge_thres:
                converged = True
                self.probabilities = t
            else:
                perplexity_old = perplexity

            iteration += 1
            print
            # for key, value in t.iteritems():
            #     print key, value


if __name__ == '__main__':
    #p_corp = [(['blue', 'house'], ['maison', 'bleu']), (['house'], ['maison'])]
    #p_corp = [(['the', 'house'], ['das', 'haus']), (['the', 'book'], ['das', 'buch']), (['a', 'book'], ['ein', 'buch'])]

    p_corp = []
    with open('corpus.en', 'rb') as corpus_en:
        with open('corpus.nl', 'rb') as corpus_nl:
            for line_en, line_nl in zip(corpus_en.readlines(), corpus_nl.readlines()):
                p_corp.append((line_en.split(), line_nl.split()))
                if len(p_corp) == 30:
                    break

    p_sentences = []
    for sentence in p_corp:
        p_sentences.append(Pair_sent(sentence))

    #for p_sent in p_sentences:
    #    print p_sent.words_e, p_sent.words_f

    ibm1 = IBM1(p_sentences, 1e-1)
    ibm1.train()

    with open('IBM1_trained.pickle', 'wb') as handle:
        pickle.dump(ibm1, handle)

    key = ('this', 'deze')
    print key, ibm1.probabilities[ibm1.dict_e[key[0]], ibm1.dict_f[key[1]]]
    # key2 = ('these', 'deze')
    # print key2, ibm1.probabilities[key2]
    # key3 = ('transparency', 'transparantie')
    # print key3, ibm1.probabilities[key3]
    #for key, value in ibm1.probabilities.iteritems():
    #    print key, value
