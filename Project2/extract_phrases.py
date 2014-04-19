from __future__ import division
import numpy as np


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

                    al = np.zeros((len(w_nl), len(w_en)), dtype=int)
                    al[w_al[:, 0], w_al[:, 1]] = 1
                    aligned_sent.append(AlignedSentences(w_en, w_nl, al))
                    so_far += 1
                    if so_far == how_many:
                        break
    return aligned_sent


if __name__ == '__main__':

    folder = 'training/'
    en_corp = 'p2_training.en'
    nl_corp = 'p2_training.nl'
    al_corp = 'p2_training_symal.nlen'
    how_many = 10
    aligned_sent = parse_aligned_sent(folder+en_corp, folder+nl_corp, folder+al_corp, how_many)

    for sent in aligned_sent:
        print sent.w_en
        print sent.w_nl
        print sent.al
        print
