from __future__ import division
import numpy as np
from collections import defaultdict
import dill as pickle


class AlignedSentences(object):

    def __init__(self, w_en, w_nl, al):
        self.w_en = w_en
        self.w_nl = w_nl
        self.al = al


def parse_aligned_sent(path_en, path_nl, path_al, how_many):
    so_far = 0
    aligned_sent = []
    with open(path_en, 'rb') as en:
        with open(path_nl, 'rb') as nl:
            with open(path_al, 'rb') as al:
                for s_en, s_nl, s_al in zip(en.readlines(), nl.readlines(), al.readlines()):
                    w_en, w_nl = s_en.split(), s_nl.split()
                    w_al = np.asarray([map(int, k.split('-')) for k in s_al.split()])

                    # ali = np.zeros((len(w_nl), len(w_en)), dtype=int)
                    # ali[w_al[:, 0], w_al[:, 1]] = 1
                    aligned_sent.append(AlignedSentences(w_en, w_nl, w_al))

                    so_far += 1
                    if how_many is not 'all':
                        if so_far == how_many:
                            break

    return aligned_sent


def parse_phrases(aligned_sent, max_len=4, saving=False, folder=None):
    print 'Parsing phrases...'
    phrase_pairs = set()
    # how to estimate the joint probability?
    # number of occurences of the phrase, divided by the number of sentences?
    # probably divided by the number of phrases
    joint_ennl = defaultdict(float)
    nl_given_en = defaultdict(lambda: defaultdict(float))
    en_given_nl = defaultdict(lambda: defaultdict(float))
    cnt = 1
    for sent in aligned_sent:
        print cnt, 'out of', len(aligned_sent)
        aligned = defaultdict(lambda: False)
        for e_start in xrange(len(sent.w_nl)):
            for e_end in xrange(e_start, e_start + max_len):
                f_start, f_end = len(sent.w_en), -1
                for e, f in sent.al:
                    aligned[f] = True
                    if e_start <= e and e <= e_end:
                        f_start = min(f, f_start)
                        f_end = max(f, f_end)
                # print 'f_start', f_start, 'f_end', f_end, 'e_start', e_start, 'e_end', e_end
                # E = extract(f_start, f_end, e_start, e_end, sent.al, sent.w_en, sent.w_nl)
                E = extract(f_start, f_end, e_start, e_end, sent.al, sent.w_en, sent.w_nl, aligned)
                for elem in E:
                    joint_ennl[elem] += 1
                    nl_given_en[elem[1]][elem[0]] += 1
                    en_given_nl[elem[0]][elem[1]] += 1

                phrase_pairs.update(E)
        cnt += 1

    # normalize the conditionals
    for key in en_given_nl:
        denom_en = sum([en_given_nl[key][key2] for key2 in en_given_nl[key]])
        for key2 in en_given_nl[key]:
            en_given_nl[key][key2] /= denom_en

    for key in nl_given_en:
        denom_nl = sum([nl_given_en[key][key2] for key2 in nl_given_en[key]])
        for key2 in nl_given_en[key]:
            nl_given_en[key][key2] /= denom_nl

    # normalize the joint
    for key in joint_ennl:
        joint_ennl[key] /= len(phrase_pairs)

    if saving:
        save_phrases(phrase_pairs, en_given_nl, nl_given_en, joint_ennl, folder)

    return phrase_pairs, en_given_nl, nl_given_en, joint_ennl


# def extract(f_start, f_end, e_start, e_end, w_a, w_en, w_nl):
def extract(f_start, f_end, e_start, e_end, w_a, w_en, w_nl, aligned):
    # print f_start, f_end, e_start, e_end
    # aligned_fe = defaultdict(lambda: False)
    # aligned_fs = defaultdict(lambda: False)
    # print
    # print 'f_start', f_start, 'f_end', f_end, 'e_start', e_start, 'e_end', e_end
    if f_end == -1:
        # print 'zero f_end'
        return set()
    for e, f in w_a:
        if f >= f_start and f <= f_end:
            if e < e_start or e > e_end:
                # print 'violating consistency'
                return set()
    E = set()
    f_s = f_start
    # print 'passed and adding possible phrases'
    while True:
        f_e = f_end
        while True:
            #if abs(e_start - e_end) <= 3:
            E.add((' '.join(w_nl[e_start:e_end+1]), ' '.join(w_en[f_s:f_e+1])))
            # aligned_fe[f_e] = True
            f_e += 1
            # if aligned_fe[f_e] or f_e == len(w_en):
            if aligned[f_e] or f_e == len(w_en):
                break
        # aligned_fs[f_s] = True
        f_s -= 1
        # if aligned_fs[f_s] or f_s < 0:
        if aligned[f_s] or f_s < 0:
            break

    return E


def save_phrases(phrase_pairs, en_given_nl, nl_given_en, joint_ennl, folder):
    print 'Saving...'
    with open(folder+'phrase_pairs.pickle', 'wb') as handle:
        pickle.dump(phrase_pairs, handle)
    with open(folder+'en_given_nl.pickle', 'wb') as handle:
        pickle.dump(en_given_nl, handle)
    with open(folder+'nl_given_en.pickle', 'wb') as handle:
        pickle.dump(nl_given_en, handle)
    with open(folder+'joint_ennl.pickle', 'wb') as handle:
        pickle.dump(joint_ennl, handle)
    print 'Saved.'


def load_phrases(folder):
    print 'Loading...'
    with open(folder+'phrase_pairs.pickle', 'rb') as handle:
        phrase_pairs = pickle.load(handle)
    with open(folder+'en_given_nl.pickle', 'rb') as handle:
        en_given_nl = pickle.load(handle)
    with open(folder+'nl_given_en.pickle', 'rb') as handle:
        nl_given_en = pickle.load(handle)
    with open(folder+'joint_ennl.pickle', 'rb') as handle:
        joint_ennl = pickle.load(handle)
    print 'Loaded.'

    return phrase_pairs, en_given_nl, nl_given_en, joint_ennl

if __name__ == '__main__':

    folder = 'training/'
    en_corp = 'p2_training.en'
    nl_corp = 'p2_training.nl'
    al_corp = 'p2_training_symal.nlen'
    how_many = 'all'
    max_len = 4
    aligned_sent = parse_aligned_sent(folder+en_corp, folder+nl_corp, folder+al_corp, how_many)
    phrase_pairs, en_given_nl, nl_given_en, joint_ennl = parse_phrases(aligned_sent, max_len=max_len, saving=False, folder=folder)

    # print 'P(en|nl)'
    # for key in en_given_nl:
    #     for key2 in en_given_nl[key]:
    #         print key, '#', key2, ':', en_given_nl[key][key2]

    # print
    # print 'P(nl|en)'
    # for key in nl_given_en:
    #     for key2 in nl_given_en[key]:
    #         print key, '#', key2, ':', nl_given_en[key][key2]

    # print
    # print 'P(en, nl)'
    # for key in joint_ennl:
    #     print key, ':', joint_ennl[key]
