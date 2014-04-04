from __future__ import division
from collections import defaultdict
import numpy as np
import dill as pickle
import os,os.path,UserDict
from sqlite3 import dbapi2 as sqlite


class dbdict(UserDict.DictMixin):
    ''' dbdict, a dictionnary-like object for large datasets (several Tera-bytes) '''

    def __init__(self, dictName):
        self.db_filename = "dbdict_%s.sqlite" % dictName
        if not os.path.isfile(self.db_filename):
            self.con = sqlite.connect(self.db_filename)
            self.con.execute("create table data (key PRIMARY KEY,value)")
        else:
            self.con = sqlite.connect(self.db_filename)

    def __getitem__(self, key):
        row = self.con.execute("select value from data where key=?",(key,)).fetchone()
        if not row: raise KeyError
        return row[0]

    def __setitem__(self, key, item):
        if self.con.execute("select key from data where key=?",(key,)).fetchone():
            self.con.execute("update data set value=? where key=?",(item,key))
        else:
            self.con.execute("insert into data (key,value) values (?,?)",(key, item))
        self.con.commit()

    def __delitem__(self, key):
        if self.con.execute("select key from data where key=?",(key,)).fetchone():
            self.con.execute("delete from data where key=?",(key,))
            self.con.commit()
        else:
             raise KeyError

    def keys(self):
        return [row[0] for row in self.con.execute("select key from data").fetchall()]


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

    def _generate_voc(self):
        self.voc_e = set()
        self.voc_f = set()

        for sent in self.p_sentences:
            self.voc_e.update(sent.words_e)
            self.voc_f.update(sent.words_f)

        self.voc_f.add(None)

    def train(self):
        t = dbdict("prob_table")
        # t = defaultdict(lambda: 1.0/len(self.voc_f))
        # t = {}
        for e in self.voc_e:
            for f in self.voc_f:
                t[str((e, f))] = 1.0/len(self.voc_f)
        print 'Finished creating.'
        #print 'Initial probabilities: %f' % (1.0/len(self.voc_f))
        converged = False
        iteration = 0
        perplexity_old = 10**200
        count = defaultdict(float)
        total = defaultdict(float)

        while not converged:
            print 'EM iteration %i' % iteration
            # init count(e|f) and total(f)
            count.clear()
            total.clear()
            # for every pair of sentences in the parallel corpus
            # gather counts
            # E - Step
            for sent in self.p_sentences:
                # total_s = {}
                total_s = {key: value for (key, value) in zip(sent.words_e, [sum([t[str((e, f))] for f in sent.words_f+[None]]) for e in sent.words_e])}
                # for e in sent.words_e:
                #     total_s[e] = sum([t[e, f] for f in sent.words_f+[None]])
                for e in sent.words_e:
                    for f in sent.words_f+[None]:
                        count[e, f] += t[str((e, f))]/total_s[e]
                        total[f] += t[str((e, f))]/total_s[e]

            # normalize and get new t(e|f)
            # M - Step
            # t = {key: value for (key, value) in zip([(e, f) for f in self.voc_f for e in self.voc_e], [count[e, f]/total[f] for f in self.voc_f for e in self.voc_e])}
            for f in self.voc_f:
                for e in self.voc_e:
                    t[str((e, f))] = count[e, f] / total[f]

            # have we converged?
            perplexity = 0
            for sent in self.p_sentences:
                mult = 1
                norm = 1/((len(sent.words_f) + 1) ** len(sent.words_e))
                for e in sent.words_e:
                    p_ = sum([t[str((e, f))] for f in sent.words_f+[None]])
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

    with open('IBM1.pickle', 'wb') as handle:
        pickle.dump(ibm1, handle)

    key = ('this', 'deze')
    print key, ibm1.probabilities[key]
    key2 = ('these', 'deze')
    print key2, ibm1.probabilities[key2]
    key3 = ('transparency', 'transparantie')
    print key3, ibm1.probabilities[key3]
    #for key, value in ibm1.probabilities.iteritems():
    #    print key, value
