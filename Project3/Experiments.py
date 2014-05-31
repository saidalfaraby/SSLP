from __future__ import division
import numpy as np
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
from KLIEP import KLIEP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from FeatureExtraction import Features
import dill as pickle
from w2vec_model import W2V


def experiment_w2vSVM(type_d, type_pooling='max_p'):
    if type_d is 'legal':
        domain = 'legal.half.en'
        domain_out = 'out.mixed.legal.en'
        model = W2V('legal', '', '')
        model.load('legal')
    elif type_d is 'software':
        domain = 'software.half.en'
        domain_out = 'out.mixed.software.en'
        model = W2V('software', '', '')
        model.load('software')

    with open('project3_data_selection/'+domain, 'rb') as doc:
        uni_sentences = [sentence.split() for sentence in doc.readlines()]

    print 'Getting training data...'
    train_data = model.create_dataset(uni_sentences, type_pooling=type_pooling)

    print 'Fitting one class SVM'
    clf = svm.OneClassSVM(kernel='linear')
    clf.fit(train_data)

    print 'Getting testing data...'
    with open('project3_data_selection/'+domain_out, 'rb') as doc:
        uni_sentences_out = [sentence.split() for sentence in doc.readlines()]

    labels = - np.ones(len(uni_sentences_out))
    labels[-50000:] = 1
    test_data = model.create_dataset(uni_sentences_out, type_pooling=type_pooling)

    print 'Predicting for test data...'
    predictions = clf.predict(test_data)

    print np.where(predictions == 1)
    p, r, f, s = precision_recall_fscore_support(labels.astype(int), predictions.astype(int), pos_label=1, average='micro')
    print 'Precision:', p,
    print 'Recall:', r,
    print 'F1:', f,
    print 'Support:', s,


def experiment_w2vKLIEP(type_d, type_pooling='max_p'):
    if type_d is 'legal':
        domain = 'legal.half.en'
        domain_out = 'out.mixed.legal.en'
        model = W2V('legal', '', '')
        model.load('legal')
    elif type_d is 'software':
        domain = 'software.half.en'
        domain_out = 'out.mixed.software.en'
        model = W2V('software', '', '')
        model.load('software')

    with open('project3_data_selection/'+domain, 'rb') as doc:
        uni_sentences = [sentence.split() for sentence in doc.readlines()]

    print 'Getting training data...'
    train_data = model.create_dataset(uni_sentences, type_pooling=type_pooling)
    print 'parsed size:', train_data.shape

    print 'Getting testing data...'
    with open('project3_data_selection/'+domain_out, 'rb') as doc:
        uni_sentences_out = [sentence.split() for sentence in doc.readlines()]

    labels = - np.ones(len(uni_sentences_out))
    labels[-50000:] = 1
    test_data = model.create_dataset(uni_sentences_out, type_pooling=type_pooling)
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
    print 'Precision:', p,
    print 'Recall:', r,
    print 'F1:', f,
    print 'Support:', s,


def experiment_tfidfKLIEP(type_d):
    if type_d is 'legal':
        domain = 'legal.half.en'
        domain_out = 'out.mixed.legal.en'
    elif type_d is 'software':
        domain = 'software.half.en'
        domain_out = 'out.mixed.software.en'

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
    with open('project3_data_selection/'+domain, 'rb') as doc:
        uni_sentences = [sentence for sentence in doc.readlines()]

    in_sent = vectorizer.fit_transform(uni_sentences)
    tsvd = TruncatedSVD(n_components=100)
    t_in_sent = tsvd.fit_transform(in_sent)
    # mm_scale = MinMaxScaler()
    # t_in_sent = mm_scale.fit_transform(t_in_sent)
    print 'parsed size:', t_in_sent.shape

    print 'Getting testing data...'
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
    print 'Precision:', p,
    print 'Recall:', r,
    print 'F1:', f,
    print 'Support:', s,


def construct_LMfeatVec(type_d, parse=False):
    if type_d is 'legal':
        domain = 'legal.half.en'
        domain_out = 'out.mixed.legal.en'
    elif type_d is 'software':
        domain = 'software.half.en'
        domain_out = 'out.mixed.software.en'

    if parse:
        In = Features()
        In.parse_doc('project3_data_selection/'+domain)
        In.save('lms_in_domain_'+type_d+'.pickle')

        Out = Features()
        Out.parse_doc('project3_data_selection/'+domain_out)
        Out.save('lms_out_domain_'+type_d+'.pickle')
    else:
        In = Features()
        In.load('lms_in_domain_'+type_d+'.pickle')

        Out = Features()
        Out.load('lms_out_domain_'+type_d+'.pickle')

    print 'Getting training data...'
    with open('project3_data_selection/'+domain, 'rb') as doc:
        uni_sentences = [sentence for sentence in doc.readlines()]

    print 'Getting testing data...'
    with open('project3_data_selection/'+domain_out, 'rb') as doc:
        uni_sentences_out = [sentence for sentence in doc.readlines()]

    save_in = 'feat_vec_in.'+type_d+'.en.pickle'

    in_d = In.construct_features(uni_sentences, use_smoothing=False)
    with open(save_in, 'wb') as handle:
        pickle.dump(in_d, handle)

    save_out = 'feat_vec_out.'+type_d+'.en.pickle'
    out_d = Out.construct_features(uni_sentences_out, use_smoothing=False)
    with open(save_out, 'wb') as handle:
        pickle.dump(out_d, handle)

    print in_d.shape, out_d.shape


def experiment_LMKLIEP(which_d):
    with open('feat_vec_out.'+which_d+'legal.en.pickle', 'rb') as handle:
        out_d = pickle.load(handle)

    with open('feat_vec_in.'+which_d+'.en.pickle', 'rb') as handle:
        in_d = pickle.load(handle)

    labels = - np.ones(out_d.shape[0])
    predictions = - np.ones(out_d.shape[0])
    labels[-50000:] = 1

    kliep = KLIEP(init_b=100, seed=0)
    kliep.fit_CV(out_d, in_d)

    w = kliep.predict(out_d).ravel()
    # predictions[np.where(w > 2.5)[0]] = 1   # w = p_te/p_tr
    # print 'total positive:', np.where(predictions == 1)[0].shape, ', out of:', test_data.shape[0]
    sorted_ind = np.argsort(w, axis=None)  # [::-1]
    predictions[sorted_ind[0:50000]] = 1

    p, r, f, s = precision_recall_fscore_support(labels.astype(int), predictions.astype(int), pos_label=1, average='micro')
    print 'Precision:', p,
    print 'Recall:', r,
    print 'F1:', f,
    print 'Support:', s,


if __name__ == '__main__':
    # experiment_w2vSVM('legal', type_pooling='max_p')
    experiment_w2vKLIEP('legal', type_pooling='max_p')
    # experiment_tfidfKLIEP('legal')
    # construct_LMfeatVec('legal', parse=False)
    # experiment_LMKLIEP('legal')
