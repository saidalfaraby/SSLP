from __future__ import division
import logging
import numpy as np
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
from KLIEP import KLIEP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from FeatureExtraction import Features
import dill as pickle


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def create_w2v_model(size):
    from gensim.models import Word2Vec
    domain = 'legal.half.en'
    with open('project3_data_selection/'+domain, 'rb') as doc:
        uni_sentences = [sentence.split() for sentence in doc.readlines()]

    domain_out = 'out.mixed.legal.en'
    with open('project3_data_selection/'+domain_out, 'rb') as doc:
        uni_sentences_out = [sentence.split() for sentence in doc.readlines()]

    model = Word2Vec(uni_sentences+uni_sentences_out, min_count=1, workers=4, size=size)
    model.save('w2vec_legal')


def main3():
    from gensim.models import Word2Vec
    from utils import create_w2v_dataset

    domain = 'legal.half.en'
    with open('project3_data_selection/'+domain, 'rb') as doc:
        uni_sentences = [sentence.split() for sentence in doc.readlines()]
    model = Word2Vec.load('w2vec_legal')

    print 'Getting training data...'
    train_data = create_w2v_dataset(uni_sentences, model)

    print 'Fitting one class SVM'
    clf = svm.OneClassSVM(kernel='linear')
    clf.fit(train_data)

    print 'Getting testing data...'
    domain_out = 'out.mixed.legal.en'
    with open('project3_data_selection/'+domain_out, 'rb') as doc:
        uni_sentences_out = [sentence.split() for sentence in doc.readlines()]

    labels = - np.ones(len(uni_sentences_out))
    labels[-50000:] = 1
    test_data = create_w2v_dataset(uni_sentences_out, model)

    print 'Predicting for test data...'
    predictions = clf.predict(test_data)

    print np.where(predictions == 1)
    p, r, f, s = precision_recall_fscore_support(labels.astype(int), predictions.astype(int), pos_label=1, average='micro')
    print 'Precision:', p
    print 'Recall:', r
    print 'F1:', f
    print 'Support:', s


def main4():
    from gensim.models import Word2Vec
    from utils import create_w2v_dataset

    domain = 'legal.half.en'
    with open('project3_data_selection/'+domain, 'rb') as doc:
        uni_sentences = [sentence.split() for sentence in doc.readlines()]
    model = Word2Vec.load('w2vec_legal')

    print 'Getting training data...'
    train_data = create_w2v_dataset(uni_sentences, model)
    print 'parsed size:', train_data.shape

    print 'Getting testing data...'
    domain_out = 'out.mixed.legal.en'
    with open('project3_data_selection/'+domain_out, 'rb') as doc:
        uni_sentences_out = [sentence.split() for sentence in doc.readlines()]

    labels = - np.ones(len(uni_sentences_out))
    labels[-50000:] = 1
    test_data = create_w2v_dataset(uni_sentences_out, model)
    print 'parsed size:', test_data.shape

    kliep = KLIEP(init_b=100, seed=0)
    kliep.fit_CV(test_data, train_data)
    predictions = - np.ones(len(uni_sentences_out))
    w = kliep.predict(test_data).ravel()
    # predictions[np.where(w > 2.5)[0]] = 1   # w = p_te/p_tr
    # print 'total positive:', np.where(predictions == 1)[0].shape, ', out of:', test_data.shape[0]
    sorted_ind = np.argsort(w, axis=None)[::-1]
    predictions[sorted_ind[0:50000]] = 1

    p, r, f, s = precision_recall_fscore_support(labels.astype(int), predictions.astype(int), pos_label=1, average='micro')
    print 'Precision:', p
    print 'Recall:', r
    print 'F1:', f
    print 'Support:', s


def main5():
    vectorizer = TfidfVectorizer(min_df=2,
    ngram_range=(1, 2),
    stop_words='english',
    strip_accents='unicode',
    norm='l2')

    vectorizer_out = TfidfVectorizer(min_df=2,
    ngram_range=(1, 2),
    stop_words='english',
    strip_accents='unicode',
    norm='l2')

    print 'Getting training data...'
    domain = 'legal.half.en'
    with open('project3_data_selection/'+domain, 'rb') as doc:
        uni_sentences = [sentence for sentence in doc.readlines()]

    in_sent = vectorizer.fit_transform(uni_sentences)
    tsvd = TruncatedSVD(n_components=100)
    t_in_sent = tsvd.fit_transform(in_sent)
    # mm_scale = MinMaxScaler()
    # t_in_sent = mm_scale.fit_transform(t_in_sent)
    print 'parsed size:', t_in_sent.shape

    print 'Getting testing data...'
    domain_out = 'out.mixed.legal.en'
    with open('project3_data_selection/'+domain_out, 'rb') as doc:
        uni_sentences_out = [sentence for sentence in doc.readlines()]

    labels = - np.ones(len(uni_sentences_out))
    labels[-50000:] = 1
    out_sent = vectorizer_out.fit_transform(uni_sentences_out)
    tsvd_out = TruncatedSVD(n_components=100)
    t_out_sent = tsvd_out.fit_transform(out_sent)
    # t_out_sent = mm_scale.transform(t_out_sent)
    # print np.where(t_out_sent < 0)
    print 'parsed size:', t_out_sent.shape

    kliep = KLIEP(init_b=100, seed=0)
    kliep.fit_CV(t_in_sent, t_out_sent)
    predictions = - np.ones(len(uni_sentences_out))
    w = kliep.predict(t_out_sent).ravel()
    # predictions[np.where(w > 2.5)[0]] = 1   # w = p_te/p_tr
    # print 'total positive:', np.where(predictions == 1)[0].shape, ', out of:', test_data.shape[0]
    sorted_ind = np.argsort(w, axis=None)[::-1]
    predictions[sorted_ind[0:50000]] = 1

    p, r, f, s = precision_recall_fscore_support(labels.astype(int), predictions.astype(int), pos_label=1, average='micro')
    print 'Precision:', p
    print 'Recall:', r
    print 'F1:', f
    print 'Support:', s


def main6():
    F = Features()
    F.parse_doc('project3_data_selection/legal.half.en')
    F.save('lms_in_domain.pickle')

    F2 = Features()
    F2.parse_doc('project3_data_selection/out.mixed.legal.en')
    F2.save('lms_out_domain.pickle')

    # In = Features()
    # In.load('lms_in_domain.pickle')

    # Out = Features()
    # Out.load('lms_out_domain.pickle')

    # print 'Getting training data...'
    # domain = 'legal.half.en'
    # with open('project3_data_selection/'+domain, 'rb') as doc:
    #     uni_sentences = [sentence for sentence in doc.readlines()]

    # print 'Getting testing data...'
    # domain_out = 'out.mixed.legal.en'
    # with open('project3_data_selection/'+domain_out, 'rb') as doc:
    #     uni_sentences_out = [sentence for sentence in doc.readlines()]

    # in_d = In.construct_features(uni_sentences)
    # with open('feat_vec_in.legal.en.pickle', 'rb') as handle:
    #     pickle.dump(in_d, handle)

    # out_d = Out.construct_features(uni_sentences_out)
    # with open('feat_vec_out.legal.en.pickle', 'rb') as handle:
    #     pickle.dump(out_d, handle)

    # print in_d.shape, out_d.shape

if __name__ == '__main__':
    # main2(20)
    main6()
