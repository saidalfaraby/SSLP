from __future__ import division
import extract_phrases as ep
from extract_phrases import AlignedSentences
import gzip
import dill as pickle


def measure_exact(train_phrases, heldout_phrases, max_len=4):
    sparsity = []
    train_phrases_per_len = []
    heldout_phrases_per_len = []

    for i in xrange(1, max_len+1):
        train_phrases_per_len.append({x for x in train_phrases if len(x[0].split(' ')) == i})
        heldout_phrases_per_len.append({x for x in heldout_phrases if len(x[0].split(' ')) == i})

    for i in xrange(len(train_phrases_per_len)):
        sparsity.append((len(train_phrases_per_len[i].intersection(heldout_phrases_per_len[i]))/len(heldout_phrases_per_len[i]),
                            len(train_phrases_per_len[i].difference(heldout_phrases_per_len[i]))/len(train_phrases_per_len[i]),
                            len(heldout_phrases_per_len[i].difference(train_phrases_per_len[i]))/len(heldout_phrases_per_len[i])))

    return sparsity


def pr_vs_moses_per_len(path_moses, ours, max_len=4):
    moses_phrases = set()
    with open(path_moses, 'rb') as moses:
        for line in moses.readlines():
            moses_phrases.add(tuple([elem.strip(' ') for elem in line.split('|||')[0:2]]))

    moses_per_len = []
    ours_per_len = []
    for i in xrange(1, max_len+1):
        moses_per_len.append({x for x in moses_phrases if len(x[0].split(' ')) == i})
        ours_per_len.append({x for x in ours if len(x[0].split(' ')) == i})

    precision, recall = [], []
    for i in xrange(len(moses_per_len)):
        precision.append(len(moses_per_len[i].intersection(ours_per_len[i]))/len(ours_per_len[i]))
        recall.append(len(moses_per_len[i].intersection(ours_per_len[i]))/len(moses_per_len[i]))

    return precision, recall


def pr_vs_moses(path_moses, ours):
    moses_phrases = set()
    with open(path_moses, 'rb') as moses:
        for line in moses.readlines():
            moses_phrases.add(tuple([elem.strip(' ') for elem in line.split('|||')[0:2]]))

    precision = len(moses_phrases.intersection(ours))/len(ours)
    recall = len(moses_phrases.intersection(ours))/len(moses_phrases)

    return precision, recall


def get_phrases(folder, en_corp, nl_corp, al_corp, how_many, max_len=4):
    aligned_sent_train = ep.parse_aligned_sent(folder+en_corp, folder+nl_corp, folder+al_corp, how_many)
    phrase_pairs_train, en_given_nl_train, nl_given_en_train, joint_ennl_train = ep.parse_phrases(aligned_sent_train, max_len=max_len, saving=True, folder=folder)

    return phrase_pairs_train


def test():
    folder = 'heldout/'
    en_corp = 'p2_heldout.en'
    nl_corp = 'p2_heldout.nl'
    al_corp = 'p2_heldout_symal.nlen'
    how_many = 5
    max_len = 4

    aligned_sent_held = ep.parse_aligned_sent(folder+en_corp, folder+nl_corp, folder+al_corp, how_many)
    phrase_pairs_held, en_given_nl_held, nl_given_en_held, joint_ennl_held = ep.parse_phrases(aligned_sent_held, max_len=max_len, saving=True, folder=folder)
    # phrase_pairs_held, en_given_nl_held, nl_given_en_held, joint_ennl_held = ep.load_phrases(folder)

    folder = 'training/'
    en_corp = 'p2_training.en'
    nl_corp = 'p2_training.nl'
    al_corp = 'p2_training_symal.nlen'
    aligned_sent_train = ep.parse_aligned_sent(folder+en_corp, folder+nl_corp, folder+al_corp, how_many)
    phrase_pairs_train, en_given_nl_train, nl_given_en_train, joint_ennl_train = ep.parse_phrases(aligned_sent_train, max_len=max_len, saving=True, folder=folder)
    # phrase_pairs_train, en_given_nl_train, nl_given_en_train, joint_ennl_train = ep.load_phrases(folder)

    ex_sparsity = measure_exact(phrase_pairs_train, phrase_pairs_held, max_len=max_len)

    for i in xrange(len(ex_sparsity)):
        print 'For phrases with n =', i+1
        print 'In train and heldout:', len(ex_sparsity[i][0])
        print 'In train and not in heldout:', len(ex_sparsity[i][1])
        print 'In heldout and not in train:', len(ex_sparsity[i][2])
        print


def test_2():
    max_len = 4

    print 'Loading heldout phrase pairs...'
    with open('/home/christos/SSLP/Project2/heldout/phrase_pairs_.pickle') as handle:
        phrase_pairs_held = pickle.load(handle)

    print 'Loading regular phrase pairs...'
    with open('/home/said/git/SSLP/Project2/training/phrase_pairs_.pickle') as handle:
        reg_phrase_pairs = pickle.load(handle)

    print 'Loading combined phrase pairs...'
    with open('/home/said/git/SSLP/Project2/training/combined_phrase_pairs_4_10.pickle') as handle:
        comb_phrase_pairs = pickle.load(handle)

    print 'Measuring sparsity...'
    ex_sparsity = measure_exact(reg_phrase_pairs, phrase_pairs_held, max_len=max_len)
    print 'Regular phrase pairs:'
    for i in xrange(len(ex_sparsity)):
        print 'For phrases with n =', i+1
        print 'In train and heldout:', ex_sparsity[i][0]
        print 'In train and not in heldout:', ex_sparsity[i][1]
        print 'In heldout and not in train:', ex_sparsity[i][2]
        print

    ex_sparsity = measure_exact(comb_phrase_pairs, phrase_pairs_held, max_len=max_len)
    print 'Combined phrase pairs:'
    for i in xrange(len(ex_sparsity)):
        print 'For phrases with n =', i+1
        print 'In train and heldout:', ex_sparsity[i][0]
        print 'In train and not in heldout:', ex_sparsity[i][1]
        print 'In heldout and not in train:', ex_sparsity[i][2]
        print

    print 'Calcuating precision and recall against Moses...'
    print 'PR for regular:'
    precision, recall = pr_vs_moses('phrase-table', reg_phrase_pairs)
    print 'Precision:', precision, 'Recall:', recall
    print

    print 'PR for combined:'
    precision, recall = pr_vs_moses('phrase-table', comb_phrase_pairs)
    print 'Precision:', precision, 'Recall:', recall
    print

    print 'Calcuating precision and recall against Moses per length...'
    print 'PR for regular:'
    precision, recall = pr_vs_moses_per_len('phrase-table', reg_phrase_pairs)
    for i in xrange(len(precision)):
        print 'n=', i+1, ':', 'Precision:', precision[i], 'Recall:', recall[i]
    print

    print 'PR for combined:'
    precision, recall = pr_vs_moses_per_len('phrase-table', comb_phrase_pairs)
    for i in xrange(len(precision)):
        print 'n=', i+1, ':', 'Precision:', precision[i], 'Recall:', recall[i]
    print

if __name__ == '__main__':

    test_2()
