from __future__ import division
import numpy as np
import cPickle as pickle


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
        self.dict_e = {key: value for (key, value) in zip(self.voc_e, range(len(self.voc_e)))}
        self.dict_f = {key: value for (key, value) in zip(self.voc_f, range(len(self.voc_f)))}

    def train(self):

        t = np.ones((len(self.voc_e), len(self.voc_f))) * (1.0/len(self.voc_f))

        print 'Finished creating.'
        converged = False
        iteration = 0
        perplexity_old = 10**200

        while not converged:
            print 'EM iteration %i' % iteration
            # init count(e|f) and total(f)
            count = np.zeros((len(self.voc_e), len(self.voc_f)))
            total = np.zeros(len(self.voc_f))

            # for every pair of sentences in the parallel corpus
            # gather counts
            # E - Step
            print 'Starting E-step'
            for sent in self.p_sentences:
                e_w = np.asarray([self.dict_e[e] for e in sent.words_e])
                f_w = np.asarray([self.dict_f[f] for f in sent.words_f+[None]])

                total_s = {key: value for (key, value) in zip(e_w, [np.sum(t[i, f_w]) for i in e_w])}

                # since we may have the same word twice
                # we need to do it per element
                for i in e_w:
                    for j in f_w:
                        count[i, j] += t[i, j]/total_s[i]
                        total[j] += t[i, j]/total_s[i]

            print 'Finished E-step'

            # normalize and get new t(e|f)
            # M - Step
            print 'Starting M-step'
            f_w = [self.dict_f[i] for i in self.voc_f]
            t = count / total

            print 'Finished M-step'

            # have we converged?
            perplexity = 0
            for sent in self.p_sentences:
                mult, norm = 1, 1/((len(sent.words_f) + 1) ** len(sent.words_e))
                for e in sent.words_e:
                    p_ = sum([t[self.dict_e[e], self.dict_f[f]] for f in sent.words_f+[None]])
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

    ibm1 = IBM1(p_sentences, 1e-1)
    ibm1.train()

    with open('IBM1_trained.pickle', 'wb') as handle:
        pickle.dump(ibm1, handle)

    key = ('this', 'deze')
    print key, ibm1.probabilities[ibm1.dict_e[key[0]], ibm1.dict_f[key[1]]]
    key2 = ('these', 'deze')
    print key2, ibm1.probabilities[ibm1.dict_e[key2[0]], ibm1.dict_f[key2[1]]]
    key3 = ('transparency', 'transparantie')
    print key3, ibm1.probabilities[ibm1.dict_e[key3[0]], ibm1.dict_f[key3[1]]]
