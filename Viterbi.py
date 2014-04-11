from __future__ import division
from IBM1 import IBM1, Pair_sent
import dill as pickle
import numpy as np
from collections import defaultdict
import re
import sys


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
            p = t[sent.words_e[e], sent.words_f[f]]
            if p > maxprob:
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
                        align[i] = map(int, re.findall('\d+\ | \d\d +', al[i], re.UNICODE))
                    alignments.append(align)
                else:
                    # print "line:" + str(aligned.strip('\n'))
                    # print "parsed: " + str(parsed)
                    # print 'Cannot parse sentence ' + str(sent)
                    # print
                    alignments.append(align)
            sent += 1

    return alignments


def getPR(giza, computed):
    precision, recall, mse, aer = ([], [], [], [])
    for i in xrange(len(giza)):
        p, r, err, ar = (0, 0, 0, 0)
        for g in giza[i]:
            if g is not 'prob':
                enu = set(computed[i][g]).intersection(set(giza[i][g]))
                if len(computed[i][g]) == len(giza[i][g]) == 0:
                    p += 1
                    r += 1
                else:
                    try:
                        p += len(enu)/len(computed[i][g])
                    except:
                        print 'no p'
                        print 'common ' + str(enu)
                        print 'giza ' + str(giza[i][g])
                        print 'computed ' + str(computed[i][g])
                        print
                        p += 0
                    try:
                        r += len(enu)/len(giza[i][g])
                    except:
                        print 'no r'
                        print 'common ' + str(enu)
                        print 'giza ' + str(giza[i][g])
                        print 'computed ' + str(computed[i][g])
                        print
                        r += 1
                    ar += 1 - ((2*len(enu))/ (len(computed[i][g]) + len(giza[i][g])))
            else:
                err += (computed[i][g] - giza[i][g]) ** 2

        precision.append(p/len(giza[i])), recall.append(r/len(giza[i]))
        mse.append(err/len(giza[i])), aer.append(ar/len(giza[i]))

    return precision, recall, aer, mse


if __name__ == '__main__':

    n_p_sent = 'all'
    direction = 'fe'
    if len(sys.argv) > 1:
        n_p_sent = int(sys.argv[1])
        if len(sys.argv) > 2:
            direction = sys.argv[2]

    if direction == 'fe':
        with open('IBM1_trained_fe.pickle', 'rb') as handle:
            ibm1 = pickle.load(handle)
    elif direction == 'ef':
        with open('IBM1_trained_ef.pickle', 'rb') as handle:
            ibm1 = pickle.load(handle)

    print 'Estimating alignments in direction: ' + direction

    p_corp = []
    with open('corpus.en', 'rb') as corpus_en:
        with open('corpus.nl', 'rb') as corpus_nl:
            for line_en, line_nl in zip(corpus_en.readlines(), corpus_nl.readlines()):
                if direction == 'ef':
                    p_corp.append((line_en.split(), [None]+line_nl.split()))
                elif direction == 'fe':
                    p_corp.append((line_nl.split(), [None]+line_en.split()))
                if n_p_sent is not 'all':
                    if len(p_corp) == n_p_sent:
                        break

    sentences = []
    for elem in p_corp:
        sentences.append(Pair_sent(elem))

    sentAligned = []
    f = open('viterbiAligned','w')
    print 'Viterbi aligments:'
    for sent in sentences:
        prob, path = simpleMax(ibm1, sent)
        path['prob'] = prob
        sentAligned.append(path)
        f.write(str(path))
        f.write('\n')
        # print sent.words_e, sent.words_f
        # print path
        # print prob
        #print path
        #print
    f.close()
       

    gizaAligned = parseGiza('corpus_1000_ennl_viterbi')

    precision, recall, aer, mse = getPR(gizaAligned, sentAligned)
    print 'Total Precision: ' + str(sum(precision)/len(gizaAligned))
    print 'Total Recall: ' + str(sum(recall)/len(gizaAligned))
    print 'Total AER: ' + str(sum(aer)/ len(gizaAligned))
    print 'Total MSE: ' + str(sum(mse)/len(gizaAligned))


    # print 'simple Max'
    # for sent in sentences:
    #     prob, path = simpleMax(ibm1, sent)
    #     print sent.words_e, sent.words_f
    #     print path
    #     print prob
