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


def parse_phrases(aligned_sent):
    phrase_pairs = set()
    for sent in aligned_sent:
        aligned = defaultdict(lambda: False)
        for e_start in xrange(len(sent.w_nl)):
            for e_end in xrange(e_start, len(sent.w_nl)):
                f_start, f_end = len(sent.w_en), -1
                for e, f in sent.al:
                    aligned[f] = True
                    if e_start <= e and e <= e_end:
                        f_start = min(f, f_start)
                        f_end = max(f, f_end)
                # print 'f_start', f_start, 'f_end', f_end, 'e_start', e_start, 'e_end', e_end
                # E = extract(f_start, f_end, e_start, e_end, sent.al, sent.w_en, sent.w_nl)
                E = extract(f_start, f_end, e_start, e_end, sent.al, sent.w_en, sent.w_nl, aligned)
                phrase_pairs.update(E)

    for elem in phrase_pairs:
        pair = elem.split('#')
        for i in pair:
            print i
        print
    return phrase_pairs


# def extract(f_start, f_end, e_start, e_end, w_a, w_en, w_nl):
def extract(f_start, f_end, e_start, e_end, w_a, w_en, w_nl,aligned):
    # print f_start, f_end, e_start, e_end
    # aligned_fe = defaultdict(lambda: False)
    # aligned_fs = defaultdict(lambda: False)
    # print
    # print 'f_start', f_start, 'f_end', f_end, 'e_start', e_start, 'e_end', e_end
    if f_end == -1:
        # print 'zero f_end'
        return set()
    for e, f in w_a:
        if f >= f_start and f<= f_end:
            if e < e_start or e > e_end:
                # print 'violating consistency'
                return set()
    E = set()
    f_s = f_start
    # print 'passed and adding possible phrases'
    while True:
        # flag1 = 0
        f_e = f_end
        while True:
            # flag = 0
            if abs(e_start - e_end) <= 3:
                E.add(' '.join(w_nl[e_start:e_end+1]) + ' # ' + ' '.join(w_en[f_s:f_e+1]))
            # aligned_fe[f_e] = True
            f_e += 1
            # if aligned_fe[f_e] or f_e == len(w_en):
            if aligned[f_e] or f_e == len(w_en):
                break
            # for element in E:
            #     try:
            #         if w_nl[f_e] in element:
            #             flag = 1
            #             break
            #     except:
            #         flag = 1
            #         break
            # if flag == 1:
            #     break
            # print f_e
        # aligned_fs[f_s] = True
        f_s -= 1
        # if aligned_fs[f_s] or f_s < 0:
        if aligned[f_s] or f_s < 0:
            break
        # for element in E:
        #     try:
        #         if w_nl[f_s] in element:
        #             flag1 = 1
        #             break
        #     except:
        #         flag1 = 1
        #         break
        # if flag1 == 1:
        #     break

    return E

if __name__ == '__main__':

    folder = 'training/'
    en_corp = 'p2_training.en'
    nl_corp = 'p2_training.nl'
    al_corp = 'p2_training_symal.nlen'
    how_many = 1
    aligned_sent = parse_aligned_sent(folder+en_corp, folder+nl_corp, folder+al_corp, how_many)
    parse_phrases(aligned_sent)
