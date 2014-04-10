from IBM1 import IBM1, Pair_sent
import dill as pickle
import numpy as np
from collections import defaultdict
import re


# def viterbi(ibm1, sent):
#     t = ibm1.probabilities
#     V = [{}]
#     path = {}

#     # Initialize base cases (t == 0)
#     for f in sent.words_f:
#         V[0][f] = 1.0
#         path[f] = []

#     for ie in range(1, len(sent.words_e)+1):
#         V.append({})
#         newpath = {}

#         for f in sent.words_f:
#             (prob, state) = max((V[ie-1][y0] * t[sent.words_e[ie-1], f], y0) for y0 in sent.words_f)
#             V[ie][f] = prob
#             newpath[f] = path[state] + [f]

#         # Don't need to remember the old paths
#         path = newpath

#     #print_dptable(V)
#     (prob, state) = max((V[ie][y], y) for y in sent.words_f)
#     return (prob, path[state])


def simpleMax(ibm1, sent):
    t = ibm1.probabilities
    prob = 1.0
    path = defaultdict(list)
    for f in range(len(sent.words_f)):
        path[f]

    for e in range(len(sent.words_e)):
        # print [(t[sent.words_e[e],sent.words_f[f]], f) for f in range(len(sent.words_f))]
        # newprob, state = max([(t[sent.words_e[e],sent.words_f[f]], f) for f in range(len(sent.words_f))])
        maxpos = -1
        maxprob = -1.0
        for f in range(len(sent.words_f)):
            p = t[sent.words_e[e],sent.words_f[f]]
            if p>maxprob:
                maxprob = p
                maxpos = f

        # print state
        prob *= maxprob
        # print state
        # print e
        path[maxpos]
        path[maxpos].append(e+1)
    return (prob, path)

def parseGiza(filename):
    with open(filename, 'rb') as giza:
        sent = 0
        alignments = []

        while True:
            info = giza.readline()
            source_sent = giza.readline()
            aligned = giza.readline()
            align = {}
            if not aligned:
                break
            else:
                align['prob'] = float(info.split(':', 1)[1].strip('\n'))
                string = aligned.decode('utf-8')
                parsed = re.findall('\)\ |\(\ |&quot;|&apos;s|&apos;|[,.!?:/;%]|[\w\.-]+|\(\{.*?\}\)', string, re.UNICODE)
                if len(parsed) % 2 == 0:
                    words, al = ([], [])
                    for i in xrange(0, len(parsed), 2):
                        words.append(parsed[i])
                        al.append(parsed[i+1])
                    for i in xrange(len(words)):
                        num = re.findall('\d+\ | \d\d +', al[i], re.UNICODE)
                        align[i] = map(int, num)
                    alignments.append(align)
                else:
                    print "line:" + str(aligned.strip('\n'))
                    print "parsed: " + str(parsed)
                    print 'Cannot parse sentence ' + str(sent)
                    print
                    alignments.append(align)
            sent += 1

    return alignments

if __name__ == '__main__':

    with open('IBM1_trained_fe.pickle', 'rb') as handle:
        ibm1 = pickle.load(handle)

    # ibm1 = IBM1()

    # with open('en_2id.pickle', 'rb') as handle:
    #     with open('nl_2id.pickle', 'rb') as handle2:
    #         ibm1.set_w2id(pickle.load(handle), pickle.load(handle2))

    # ibm1.set_probabilities(np.loadtxt('trained_ibm1.txt', delimiter=','))

    # key3 = ('transparency', 'transparantie')
    # print key3, ibm1.probabilities[key3]

    p_corp = []
    with open('corpus.en', 'rb') as corpus_en:
        with open('corpus.nl', 'rb') as corpus_nl:
            for line_en, line_nl in zip(corpus_en.readlines(), corpus_nl.readlines()):
                p_corp.append((line_nl.split(), [None]+line_en.split()))
                if len(p_corp) == 2:
                    break

    sentences = []
    for elem in p_corp:
        sentences.append(Pair_sent(elem))

        

    sentAligned = []
    print 'viterbi'
    for sent in sentences:
        prob, path = simpleMax(ibm1, sent)
        path['prob'] = prob
        sentAligned.append(path)
        # print sent.words_e, sent.words_f
        # print path
        # print prob
        print path
        print

    gizaAligned = parseGiza('corpus_1000_ennl_viterbi')
    

    # print 'simple Max'
    # for sent in sentences:
    #     prob, path = simpleMax(ibm1, sent)
    #     print sent.words_e, sent.words_f
    #     print path
    #     print prob
