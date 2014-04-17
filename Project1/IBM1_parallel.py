from __future__ import division
import numpy as np
import cPickle as pickle
import sys
import multiprocessing as mpc
import sharedmem as shm


# helper function for pickling and parallelizing
def unwrap_self_f(arg, **kwarg):
    return IBM1.gather_counts(*arg, **kwarg)


class Pair_sent(object):

    def __init__(self, sentence):
        self.words_e = sentence[0]
        self.words_f = sentence[1]


class IBM1(object):

    def __init__(self, p_sentences=None, converge_thres=1e-1, num_proc=4):
        if p_sentences is not None:
            self.p_sentences = p_sentences
            self._generate_voc()
            self._generate_dict_ind()
        self.converge_thres = converge_thres
        self.probabilities = None
        self.num_proc = num_proc

    def set_probabilities(self, probabilities):
        self.probabilities = probabilities

    def set_w2id(self, w_e_2id, w_f_2id):
        self.dict_e = w_e_2id
        self.dict_f = w_f_2id

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

    def chunks(self, l, n):
        return [l[i:i+n] for i in range(0, len(l), n)]

    def gather_counts(self, sentences, t):

        # t = np.ctypeslib.as_array(t_ctypes)
        # t.shape = shape

        count = np.zeros((len(self.voc_e), len(self.voc_f)))
        total = np.zeros(len(self.voc_f))

        for sent in self.p_sentences:
            e_w = np.asarray([self.dict_e[e] for e in sent.words_e])
            f_w = np.asarray([self.dict_f[f] for f in sent.words_f+[None]])

            total_s = {key: value for (key, value) in zip(e_w, [np.sum(t[i, f_w]) for i in e_w])}

            for i in e_w:
                # use bincount to find duplicates and do the
                # assignments accordingly
                f_w_bin = np.bincount(f_w)
                unique = np.unique(f_w)
                f_w_bin = f_w_bin[unique]
                count[i, unique] += f_w_bin * t[i, unique]/total_s[i]
                total[unique] += f_w_bin * t[i, unique]/total_s[i]

        return count, total

    def train(self):

        t = shm.empty((len(self.voc_e), len(self.voc_f)), np.float64)
        t[:] = np.ones((len(self.voc_e), len(self.voc_f))) * (1.0/len(self.voc_f))
        # t = (shm.zeros((len(self.voc_e), len(self.voc_f))) + 1) * (1.0/len(self.voc_f))

        # size = t.size
        # shape = t.shape
        # t.shape = size
        # t_ctypes = mpc.sharedctypes.RawArray('d', t)
        # t = np.frombuffer(t_ctypes, dtype=np.float64, count=size)
        # t.shape = shape

        print 'Finished creating.'
        converged = False
        iteration = 0
        perplexity_old = 10**200

        pool = mpc.Pool()
        while not converged:
            print 'EM iteration %i' % iteration
            # init count(e|f) and total(f)
            count = np.zeros((len(self.voc_e), len(self.voc_f)))
            total = np.zeros(len(self.voc_f))

            # for every pair of sentences in the parallel corpus
            # gather counts
            # E - Step
            print 'E-step...'

            chunks = self.chunks(p_sentences, int(len(p_sentences)/self.num_proc))
            inputs = [(self, chunk, t) for chunk in chunks]

            for frac_count, frac_total in pool.map(unwrap_self_f, inputs):
                count += frac_count
                total += frac_total

            # normalize and get new t(e|f)
            # M - Step
            print 'M-step...'
            t = count / total

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
                self.set_probabilities(t)
                print 'Model converged.'
            else:
                perplexity_old = perplexity

            iteration += 1
            print


if __name__ == '__main__':
    #p_corp = [(['blue', 'house'], ['maison', 'bleu']), (['house'], ['maison'])]
    #p_corp = [(['the', 'house'], ['das', 'haus']), (['the', 'book'], ['das', 'buch']), (['a', 'book'], ['ein', 'buch'])]
    n_p_sent = 'all'
    if len(sys.argv) > 1:
        n_p_sent = int(sys.argv[1])

    print 'Training for ' + str(n_p_sent) + ' sentences...'
    p_corp = []
    with open('corpus.en', 'rb') as corpus_en:
        with open('corpus.nl', 'rb') as corpus_nl:
            for line_en, line_nl in zip(corpus_en.readlines(), corpus_nl.readlines()):
                p_corp.append((line_en.split(), line_nl.split()))
                if n_p_sent is not 'all':
                    if len(p_corp) == n_p_sent:
                        break

    p_sentences = []
    for sentence in p_corp:
        p_sentences.append(Pair_sent(sentence))

    ibm1 = IBM1(p_sentences=p_sentences, converge_thres=1e-1, num_proc=4)
    ibm1.train()

    # # save the model
    # np.savetxt('trained_ibm1.txt', ibm1.probabilities, delimiter=',')
    # with open('en_2id.pickle', 'wb') as handle:
    #     pickle.dump(ibm1.dict_e, handle)
    # with open('nl_2id.pickle', 'wb') as handle:
    #     pickle.dump(ibm1.dict_f, handle)

    print 'Saving the model to disk...'
    with open('IBM1_trained.pickle', 'wb') as handle:
        pickle.dump(ibm1, handle)

    key = ('this', 'deze')
    print key, ibm1.probabilities[ibm1.dict_e[key[0]], ibm1.dict_f[key[1]]]
    key2 = ('these', 'deze')
    print key2, ibm1.probabilities[ibm1.dict_e[key2[0]], ibm1.dict_f[key2[1]]]
    key3 = ('transparency', 'transparantie')
    print key3, ibm1.probabilities[ibm1.dict_e[key3[0]], ibm1.dict_f[key3[1]]]
