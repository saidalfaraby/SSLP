from __future__ import division
from gensim.models import Word2Vec
import numpy as np
# import logging

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class W2V(object):

    def __init__(self, type_domain, mixed_domain, in_domain, size=10):
        self.type_domain = type_domain
        self.mixed_domain = mixed_domain
        self.in_domain = in_domain
        self.size = size

    def fit(self):
        with open('project3_data_selection/'+self.in_domain, 'rb') as doc:
            uni_sentences = [sentence.split() for sentence in doc.readlines()]

        with open('project3_data_selection/'+self.mixed_domain, 'rb') as doc:
            uni_sentences_out = [sentence.split() for sentence in doc.readlines()]

        self.model = Word2Vec(uni_sentences+uni_sentences_out, min_count=1, workers=4, size=self.size)
        self.model.save('w2vec_'+self.type_domain+'.model')

    def create_dataset(self, sentences, type_pooling='average_p'):
        data = []
        for sent in sentences:
            feat_vec = []
            for w in sent:
                feat_vec.append(self.model[w])
            if type_pooling is 'average_p':
                data.append(sum(feat_vec)/len(sent))
            elif type_pooling is 'max_p':
                tmp = np.asarray(feat_vec)
                data.append(np.amax(tmp, axis=0))
        return np.asarray(data)

    def load(self, type_domain):
        self.type_domain = type_domain
        self.model = Word2Vec.load('w2vec_'+type_domain+'.model')


if __name__ == '__main__':
    w2v = W2V('legal', 'out.mixed.legal.en', 'legal.half.en', size=10)
    w2v.load('legal')
    domain = 'legal.half.en'
    with open('project3_data_selection/'+domain, 'rb') as doc:
        uni_sentences = [sentence.split() for sentence in doc.readlines()]

    data = w2v.create_dataset(uni_sentences, type_pooling='max_p')
    print data
    print data.shape
