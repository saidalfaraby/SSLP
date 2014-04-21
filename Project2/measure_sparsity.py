import extract_phrases as ep
from extract_phrases import AlignedSentences


def measure_exact(train_phrases, heldout_phrases, max_len=4):
    sparsity = []
    train_phrases_per_len = []
    heldout_phrases_per_len = []

    for i in xrange(1, max_len+1):
        train_phrases_per_len.append({x for x in train_phrases if len(x[0].split(' ')) == i})
        heldout_phrases_per_len.append({x for x in heldout_phrases if len(x[0].split(' ')) == i})

    for i in xrange(len(train_phrases_per_len)):
        sparsity.append((train_phrases_per_len[i].intersection(heldout_phrases_per_len[i]),
                            train_phrases_per_len[i].difference(heldout_phrases_per_len[i]),
                            heldout_phrases_per_len[i].difference(train_phrases_per_len[i])))

    return sparsity


if __name__ == '__main__':
    folder = 'heldout/'
    en_corp = 'p2_heldout.en'
    nl_corp = 'p2_heldout.nl'
    al_corp = 'p2_heldout_symal.nlen'
    how_many = 300 #'all'
    max_len = 4

    aligned_sent_held = ep.parse_aligned_sent(folder+en_corp, folder+nl_corp, folder+al_corp, how_many)
    phrase_pairs_held, en_given_nl_held, nl_given_en_held, joint_ennl_held = ep.parse_phrases(aligned_sent_held, max_len=max_len)

    folder = 'training/'
    en_corp = 'p2_training.en'
    nl_corp = 'p2_training.nl'
    al_corp = 'p2_training_symal.nlen'
    aligned_sent_train = ep.parse_aligned_sent(folder+en_corp, folder+nl_corp, folder+al_corp, how_many)
    phrase_pairs_train, en_given_nl_train, nl_given_en_train, joint_ennl_train = ep.parse_phrases(aligned_sent_train, max_len=max_len)

    ex_sparsity = measure_exact(phrase_pairs_train, phrase_pairs_held, max_len=max_len)

    for i in xrange(len(ex_sparsity)):
        print 'For phrases with n =', i+1
        print 'In train and heldout:', len(ex_sparsity[i][0])
        print 'In train and not in heldout:', len(ex_sparsity[i][1])
        print 'In heldout and not in train:', len(ex_sparsity[i][2])
        print
