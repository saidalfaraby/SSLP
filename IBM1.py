from collections import defaultdict


class p_sentence(object):

    def __init__(self, sentence):
        self.words_e = sentence[0]
        self.words_f = sentence[1]


class IBM1(object):

    def __init__(self, p_sentences):
        self.p_sentences = p_sentences


if __name__ == '__main__':
    p_corp = [(['blue', 'house'], ['maison', 'bleu']), (['house'], ['maison'])]

    p_sentences = []
    for sentence in p_corp:
        p_sentences.append(p_sentence(sentence))

    for p_sent in p_sentences:
        print p_sent.words_e, p_sent.words_f
