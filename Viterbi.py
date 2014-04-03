from IBM1 import IBM1, Pair_sent
import dill as pickle


def viterbi(ibm1, sent):
    t = ibm1.probabilities
    V = [{}]
    path = {}

    # Initialize base cases (t == 0)
    for f in sent.words_f:
        V[0][f] = 1.0
        path[f] = [f]

    for ie in range(1, len(sent.words_e)):
        V.append({})
        newpath = {}

        for f in sent.words_f:
            (prob, state) = max((V[ie-1][y0] * t[sent.words_e[ie], f], y0) for y0 in sent.words_f)
            V[ie][f] = prob
            newpath[f] = path[state] + [f]

        # Don't need to remember the old paths
        path = newpath

    #print_dptable(V)
    (prob, state) = max((V[ie][y], y) for y in sent.words_f)
    return (prob, path[state])

if __name__ == '__main__':

    with open('IBM1.pickle', 'rb') as handle:
        ibm1 = pickle.load(handle)

    # key3 = ('transparency', 'transparantie')
    # print key3, ibm1.probabilities[key3]

    p_corp = []
    with open('corpus.en', 'rb') as corpus_en:
        with open('corpus.nl', 'rb') as corpus_nl:
            for line_en, line_nl in zip(corpus_en.readlines(), corpus_nl.readlines()):
                p_corp.append((line_en.split(), line_nl.split()))
                if len(p_corp) == 1:
                    break

    sentences = []
    for elem in p_corp:
        sentences.append(Pair_sent(elem))

    for sent in sentences:
    	prob, path = viterbi(ibm1, sent)
    	print sent.words_e, sent.words_f
        print path
