from __future__ import division
import numpy as np
from collections import defaultdict


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
                    if so_far == how_many:
                        break

    return aligned_sent


def parse_phrases(aligned_sent, max_len=4):
    phrase_pairs = set()
    # how to estimate the joint probability?
    # number of occurences of the phrase, divided by the number of sentences?
    # joint_probs = defaultdict(float)
    nl_given_en = defaultdict(lambda: defaultdict(float))
    en_given_nl = defaultdict(lambda: defaultdict(float))
    for sent in aligned_sent:
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
                    # joint_probs[elem] += 1/len(aligned_sent)
                    nl_given_en[elem[1]][elem[0]] += 1
                    en_given_nl[elem[0]][elem[1]] += 1

                phrase_pairs.update(E)

    # normalize the conditionals
    for key in en_given_nl:
        denom_en = sum([en_given_nl[key][key2] for key2 in en_given_nl[key]])
        for key2 in en_given_nl[key]:
            en_given_nl[key][key2] /= denom_en

    for key in nl_given_en:
        denom_nl = sum([nl_given_en[key][key2] for key2 in nl_given_en[key]])
        for key2 in nl_given_en[key]:
            nl_given_en[key][key2] /= denom_nl

    for key in en_given_nl:
        for key2 in en_given_nl[key]:
            print key, '#', key2, ':', en_given_nl[key][key2]

    print

    for key in nl_given_en:
        for key2 in nl_given_en[key]:
            print key, '#', key2, ':', nl_given_en[key][key2]
    # for elem in phrase_pairs:
    #     print elem[0], '#',  elem[1]
    # for key in joint_probs:
    #         print key, joint_probs[key]

    return phrase_pairs, en_given_nl, nl_given_en


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

if __name__ == '__main__':

    folder = 'training/'
    en_corp = 'p2_training.en'
    nl_corp = 'p2_training.nl'
    al_corp = 'p2_training_symal.nlen'
    how_many = 1
    max_len = 4
    aligned_sent = parse_aligned_sent(folder+en_corp, folder+nl_corp, folder+al_corp, how_many)
    parse_phrases(aligned_sent, max_len=max_len)
