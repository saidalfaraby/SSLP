from __future__ import division
import numpy as np
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
from KLIEP import KLIEP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import MinMaxScaler
from FeatureExtraction import Features
import dill as pickle
from w2vec_model import W2V


def create_W2Vmodel(type_d, size_h=10):
    w2v = W2V(type_d, 'out.mixed.'+type_d+'.en', type_d+'.half.en', size=size_h)
    w2v.fit()


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

    print 'total positive:', np.where(predictions == 1)[0].shape[0], ', out of:', test_data.shape[0]
    p, r, f, s = precision_recall_fscore_support(labels.astype(int), predictions.astype(int), pos_label=1, average='micro')
    print 'Precision:', p,
    print 'Recall:', r,
    print 'F1:', f,
    print 'Support:', s,

    return np.where(predictions == 1)[0].tolist()


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
    predictions[np.where(w > 2.)[0]] = 1   # w = p_te/p_tr

    print 'total positive:', np.where(predictions == 1)[0].shape, ', out of:', test_data.shape[0]
    # sorted_ind = np.argsort(w, axis=None)[::-1]
    # predictions[sorted_ind[0:50000]] = 1
    # print w[sorted_ind[0:50000]]

    p, r, f, s = precision_recall_fscore_support(labels.astype(int), predictions.astype(int), pos_label=1, average='micro')
    print 'Precision:', p,
    print 'Recall:', r,
    print 'F1:', f,
    print 'Support:', s,

    return np.where(predictions == 1)[0].tolist()


def experiment_tfidfSVM(type_d, topics=100, method='tsvd'):
    if type_d is 'legal':
        domain = 'legal.half.en'
        domain_out = 'out.mixed.legal.en'
    elif type_d is 'software':
        domain = 'software.half.en'
        domain_out = 'out.mixed.software.en'

    vectorizer = TfidfVectorizer(min_df=2,
    ngram_range=(1, 2),
    stop_words='english',
    binary=True,
    strip_accents='unicode',
    norm='l2')

    vectorizer_out = TfidfVectorizer(min_df=2,
    ngram_range=(1, 2),
    stop_words='english',
    binary=True,
    strip_accents='unicode',
    norm='l2')

    print 'Getting training data...'
    with open('project3_data_selection/'+domain, 'rb') as doc:
        uni_sentences = [sentence for sentence in doc.readlines()]

    in_sent = vectorizer.fit_transform(uni_sentences)
    if method is 'tsvd':
        tsvd = TruncatedSVD(n_components=topics)
    elif method is 'nmf':
        tsvd = NMF(n_components=100, sparseness='data', init='nndsvd')
    t_in_sent = tsvd.fit_transform(in_sent)
    # mm_scale = MinMaxScaler()
    # t_in_sent = mm_scale.fit_transform(t_in_sent)
    print 'parsed size:', t_in_sent.shape

    print 'Getting testing data...'
    with open('project3_data_selection/'+domain_out, 'rb') as doc:
        uni_sentences_out = [sentence for sentence in doc.readlines()]

    labels = - np.ones(len(uni_sentences_out))
    labels[-50000:] = 1
    out_sent = vectorizer.transform(uni_sentences_out)
    # tsvd_out = TruncatedSVD(n_components=20)
    t_out_sent = tsvd.transform(out_sent)
    # t_out_sent = mm_scale.transform(t_out_sent)
    # print np.where(t_out_sent < 0)
    print 'parsed size:', t_out_sent.shape

    print 'Fitting one class SVM'
    clf = svm.OneClassSVM(kernel='linear')
    clf.fit(t_in_sent)

    print 'Predicting for test data...'
    predictions = clf.predict(t_out_sent)

    print 'total positive:', np.where(predictions == 1)[0].shape, ', out of:', t_out_sent.shape[0]
    # sorted_ind = np.argsort(w, axis=None)[::-1]
    # predictions[sorted_ind[0:50000]] = 1

    p, r, f, s = precision_recall_fscore_support(labels.astype(int), predictions.astype(int), pos_label=1, average='micro')
    print 'Precision:', p,
    print 'Recall:', r,
    print 'F1:', f,
    print 'Support:', s,


def experiment_tfidfKLIEP(type_d, topics=100, method='tsvd'):
    if type_d is 'legal':
        domain = 'legal.half.en'
        domain_out = 'out.mixed.legal.en'
    elif type_d is 'software':
        domain = 'software.half.en'
        domain_out = 'out.mixed.software.en'

    vectorizer = TfidfVectorizer(min_df=2,
    ngram_range=(1, 2),
    stop_words='english',
    binary=True,
    strip_accents='unicode',
    norm='l2')

    vectorizer_out = TfidfVectorizer(min_df=2,
    ngram_range=(1, 2),
    stop_words='english',
    binary=True,
    strip_accents='unicode',
    norm='l2')

    print 'Getting training data...'
    with open('project3_data_selection/'+domain, 'rb') as doc:
        uni_sentences = [sentence for sentence in doc.readlines()]

    in_sent = vectorizer.fit_transform(uni_sentences)
    if method is 'tsvd':
        tsvd = TruncatedSVD(n_components=topics)
    elif method is 'nmf':
        tsvd = NMF(n_components=100, sparseness='data', init='nndsvd')
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
    tsvd_out = TruncatedSVD(n_components=topics)
    t_out_sent = tsvd_out.fit_transform(out_sent)
    # t_out_sent = mm_scale.transform(t_out_sent)
    # print np.where(t_out_sent < 0)
    print 'parsed size:', t_out_sent.shape

    kliep = KLIEP(init_b=100, seed=0)
    kliep.fit_CV(t_out_sent, t_in_sent)
    predictions = - np.ones(len(uni_sentences_out))
    w = kliep.predict(t_out_sent).ravel()
    predictions[np.where(w > 1.)[0]] = 1   # w = p_te/p_tr
    print 'total positive:', np.where(predictions == 1)[0].shape, ', out of:', t_out_sent.shape[0]
    # sorted_ind = np.argsort(w, axis=None)[::-1]
    # predictions[sorted_ind[0:50000]] = 1

    p, r, f, s = precision_recall_fscore_support(labels.astype(int), predictions.astype(int), pos_label=1, average='micro')
    print 'Precision:', p,
    print 'Recall:', r,
    print 'F1:', f,
    print 'Support:', s,


def construct_LMfeatVec(type_d, parse=False, prune='noprune'):
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
        # In.load('lms_in_domain_'+type_d+'.pickle')
        In.load(type_d+'_in_model_'+prune+'.pickle')

        Out = Features()
        # Out.load('lms_out_domain_'+type_d+'.pickle')
        Out.load(type_d+'_sample_mix_model_'+prune+'.pickle')

    print 'Getting training data...'
    with open('project3_data_selection/'+domain, 'rb') as doc:
        uni_sentences = [sentence for sentence in doc.readlines()]

    print 'Getting testing data...'
    with open('project3_data_selection/'+domain_out, 'rb') as doc:
        uni_sentences_out = [sentence for sentence in doc.readlines()]

    save_in = 'feat_vec_in.'+type_d+'.en.pickle'

    in_d = In.construct_features(uni_sentences, use_smoothing=True)
    with open(save_in, 'wb') as handle:
        pickle.dump(in_d, handle)

    save_out = 'feat_vec_out.'+type_d+'.en.pickle'
    out_d = Out.construct_features(uni_sentences_out, use_smoothing=True)
    with open(save_out, 'wb') as handle:
        pickle.dump(out_d, handle)

    print in_d.shape, out_d.shape


def experiment_LMKLIEP(which_d):
    with open('feat_vec_out.'+which_d+'.en.pickle', 'rb') as handle:
        out_d = pickle.load(handle)

    with open('feat_vec_in.'+which_d+'.en.pickle', 'rb') as handle:
        in_d = pickle.load(handle)

    labels = - np.ones(out_d.shape[0])
    predictions = - np.ones(out_d.shape[0])
    labels[-50000:] = 1

    kliep = KLIEP(init_b=100, seed=0)
    kliep.fit_CV(out_d, in_d)

    w = kliep.predict(out_d).ravel()
    predictions[np.where(w > 1)[0]] = 1   # w = p_te/p_tr
    print 'total positive:', np.where(predictions == 1)[0].shape, ', out of:', out_d.shape[0]
    # sorted_ind = np.argsort(w, axis=None)[::-1]
    # predictions[sorted_ind[0:50000]] = 1

    p, r, f, s = precision_recall_fscore_support(labels.astype(int), predictions.astype(int), pos_label=1, average='micro')
    print 'Precision:', p,
    print 'Recall:', r,
    print 'F1:', f,
    print 'Support:', s,


def experiment_LMSVM(which_d):
    with open('feat_vec_out.'+which_d+'.en.pickle', 'rb') as handle:
        out_d = pickle.load(handle)

    with open('feat_vec_in.'+which_d+'.en.pickle', 'rb') as handle:
        in_d = pickle.load(handle)

    labels = - np.ones(out_d.shape[0])
    labels[-50000:] = 1

    print 'Fitting one class SVM'
    clf = svm.OneClassSVM(kernel='linear')
    clf.fit(in_d)

    print 'Predicting for out domain'
    predictions = clf.predict(out_d)
    print 'total positive:', np.where(predictions == 1)[0].shape, ', out of:', out_d.shape[0]
    # sorted_ind = np.argsort(w, axis=None)[::-1]
    # predictions[sorted_ind[0:50000]] = 1

    p, r, f, s = precision_recall_fscore_support(labels.astype(int), predictions.astype(int), pos_label=1, average='micro')
    print 'Precision:', p,
    print 'Recall:', r,
    print 'F1:', f,
    print 'Support:', s,


def experimentLMKLIEP_per_feat(which_d):
    with open('feat_vec_out.'+which_d+'.en.pickle', 'rb') as handle:
        out_d = pickle.load(handle)

    with open('feat_vec_in.'+which_d+'.en.pickle', 'rb') as handle:
        in_d = pickle.load(handle)

    labels = - np.ones(out_d.shape[0])
    predictions = - np.ones(out_d.shape[0])
    labels[-50000:] = 1

    W = np.zeros(out_d.shape)

    for i in xrange(W.shape[1]):
        kliep = KLIEP(init_b=100, seed=0)
        lm_out = out_d[:, i].reshape((out_d.shape[0], 1))
        lm_in = in_d[:, i].reshape((in_d.shape[0], 1))
        kliep.fit_CV(lm_out, lm_in)
        W[:, i] = kliep.predict(lm_out).ravel()

    w = np.mean(W, axis=1)
    predictions[np.where(w > 1.4)[0]] = 1
    print 'total positive:', np.where(predictions == 1)[0].shape, ', out of:', out_d.shape[0]
    # sorted_ind = np.argsort(w, axis=None)[::-1]
    # predictions[sorted_ind[0:50000]] = 1

    p, r, f, s = precision_recall_fscore_support(labels.astype(int), predictions.astype(int), pos_label=1, average='micro')
    print 'Precision:', p,
    print 'Recall:', r,
    print 'F1:', f,
    print 'Support:', s,


def write_doc(in_filename, out_filename, in_d_filename, Selected_Docs):
    print 'writing selected docs from '+in_filename
    print 'to '+out_filename
    fo = open(out_filename, 'wb')
    fi = open(in_filename, 'rb')
    fd = open(in_d_filename, 'rb')
    doc = fi.readlines()
    in_doc = fd.readlines()
    for line in in_doc:
        fo.write(line+'\n')
    for i in Selected_Docs:
      fo.write(doc[i]+'\n')
    fo.close()
    fi.close()
    fd.close()


if __name__ == '__main__':
    # average pooling seems to work better than max pooling
    # create_W2Vmodel('software', size_h=20)
    # create_W2Vmodel('legal', size_h=20)

    docs = experiment_w2vSVM('legal', type_pooling='average_p')  # P:0.62, R: 0.49, F1: 0.55
    write_doc('project3_data_selection/out.mixed.legal.en', 'selected.legal.svm.en', 'project3_data_selection/legal.half.en', docs)
    write_doc('project3_data_selection/out.mixed.legal.es', 'selected.legal.svm.es', 'project3_data_selection/legal.half.es', docs)
    # experiment_w2vSVM('software', type_pooling='average_p')  # Precision: 0.354820175379 Recall: 0.51874 F1: 0.421400661256

    docs = experiment_w2vKLIEP('legal', type_pooling='average_p')  # Precision: 0.687512635622 Recall: 0.40808 F1: 0.512161449836
    write_doc('project3_data_selection/out.mixed.legal.en', 'selected.legal.kliep.en', 'project3_data_selection/legal.half.en', docs)
    write_doc('project3_data_selection/out.mixed.legal.es', 'selected.legal.kliep.es', 'project3_data_selection/legal.half.es', docs)
    # experiment_w2vKLIEP('software', type_pooling='average_p')  # 0.543853551153 Recall: 0.37552 F1: 0.444276181913

    # experiment_tfidfKLIEP('legal', topics=5, method='tsvd')  # Precision: 0.144105604336 Recall: 0.77006 F1: 0.242778685062
    # experiment_tfidfKLIEP('software', topics=5, method='tsvd')  # Precision: 0.0193548387097 Recall: 6e-05 F1: 0.000119629149636

    # experiment_tfidfSVM('legal', topics=100, method='tsvd')  # P: 0.32, R: 0.55, F1: 0.4
    # experiment_tfidfSVM('software', topics=100, method='tsvd')  # Precision: 0.22344679106 Recall: 0.52308 F1: 0.313131555002

    # experimentLMKLIEP_per_feat('legal')  # Precision: 0.222645099905 Recall: 0.27144 F1: 0.244633104418
    # construct_LMfeatVec('legal', parse=False)
    # construct_LMfeatVec('software', parse=False)
    # experiment_LMKLIEP('legal')  # Precision: 0.21700076066 Recall: 0.38798 F1: 0.278329363827
    # experiment_LMKLIEP('software') # Precision: 0.107654415738 Recall: 0.54986 F1: 0.180056453884
    # experiment_LMSVM('legal')  # Precision: 0.0771818727491 Recall: 0.51434 F1: 0.134222338205
    # experiment_LMSVM('software')  # Precision: 0.0963900811893 Recall: 0.45542 F1: 0.159105359875
