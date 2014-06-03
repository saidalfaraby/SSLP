from __future__ import division
from collections import defaultdict
import nltk
import re
import string
from nltk.corpus import stopwords
import dill as pickle
import numpy as np
import random


class Features(object):
  """docstring for Features"""
  def __init__(self):
    self.Term_Freq = defaultdict(int)  # term frequency. Key = term, Val = frequency
    self.POS_Freq = defaultdict(int)  # postag frequency. Key = POSTag, Val = Frequency
    self.TPOS_Freq = defaultdict(int)  # term, postag frequency. Key = (term,POSTag), Val = Frequency
    self.N_Term = 0

    self.Tr_Term_Freq = defaultdict(int) #for translation language model
    self.Tr_TPOS_Freq = defaultdict(int) #for translation language model
    self.Tr_POS_Freq = defaultdict(int)
    self.Tr_N_Term = 0



    self.BTerm_Freq = defaultdict(int)  # bigram frequency. Key = (term, term), Val = Frequency
    self.BPOS_Freq = defaultdict(int)  # bigram POS-tags. key = (POSTag, POSTag), Val = Frequency
    self.BTPOS_Freq = defaultdict(int)
    self.BN_Term = 0

    self.Trans_Freq = defaultdict(int)  # translation frequency. Key = (E,F), Val = frequency

  def parse_doc(self, path, nrandom_sample=None, is_translation=False):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    i = 0

    f = open(path)
    docs = f.readlines()
    if nrandom_sample!=None:
      newdocs = []
      for idx in random.sample(range(len(docs)),nrandom_sample):
        newdocs.append(docs[idx])
      docs = newdocs
    #print len(docs)
    #print docs
    if not is_translation:
      for sentence in docs:
        sentence = nltk.pos_tag(nltk.word_tokenize(sentence))
        i += 1
        print i
        #remove punctuation
        for token, pos in sentence:
          try:
            new_token = regex.sub(u'', token).decode('utf-8')
            if not new_token == u'' and not new_token in stopwords.words('english'):
              self.update_count(new_token, pos)
          except:
            pass

        b_words, b_pos = self.to_bigram(sentence)

        for b_w, b_p in zip(b_words, b_pos):
          self.update_count(b_w, b_p, bigrams=1)
    else :
      spanish_tagger = pickle.load(open('bitag_spanish.pickle','rb'))
      for sentence in docs:
        i += 1
        print i
        term_pos = spanish_tagger.tag(nltk.word_tokenize(sentence))
        for term,pos in term_pos :
          if not term == '' and not term in stopwords.words('spanish'):
            self.update_count(term, pos, is_translation=True)
    self.set_lambda(1)
    # self.prune()

  def to_bigram(self, termpos):
    words = [elem[0] for elem in termpos]
    pos_tags = [elem[1] for elem in termpos]

    b_words = nltk.bigrams(words)
    b_pos = nltk.bigrams(pos_tags)
    return (b_words, b_pos)

  def set_lambda(self, v):
    # unigram
    self.Term_Freq.default_factory = lambda: v
    self.TPOS_Freq.default_factory = lambda: v
    self.Trans_Freq.default_factory = lambda: v
    self.POS_Freq.default_factory = lambda: v
    self.Tr_Term_Freq.default_factory = lambda : v
    self.Tr_TPOS_Freq.default_factory = lambda : v
    self.Tr_POS_Freq.default_factory = lambda : v
    # bigram
    self.BTerm_Freq.default_factory = lambda: v
    self.BPOS_Freq.default_factory = lambda: v
    self.BTPOS_Freq.default_factory = lambda: v

  def prune_dict(self, dictionary, bigram=0):
    keys = dictionary.keys()
    n_removed =0
    for k in keys:
      if dictionary[k] <= 1:
        del dictionary[k]
        n_removed +=1
    return n_removed

  def prune(self):
    # unigram
    self.prune_dict(self.Term_Freq)
    self.prune_dict(self.POS_Freq)
    uni_removed = self.prune_dict(self.TPOS_Freq)
    self.N_Term -= uni_removed
    # bigram
    self.prune_dict(self.BTerm_Freq, bigram=1)
    self.prune_dict(self.BPOS_Freq, bigram=1)
    bi_removed = self.prune_dict(self.BTPOS_Freq, bigram=1)
    self.BN_Term -= bi_removed

  def update_count(self, t, p, bigrams=0, is_translation=False):
    if not is_translation:
      if bigrams == 0:
        self.Term_Freq[t] += 1
        self.TPOS_Freq[(t, p)] += 1
        self.N_Term += 1
        self.POS_Freq[p] += 1
      elif bigrams == 1:
        self.BN_Term += 1
        self.BTerm_Freq[t] += 1
        self.BPOS_Freq[p] += 1
        self.BTPOS_Freq[(t, p)] += 1
    else :
      if bigrams == 0:
        self.Tr_Term_Freq[t] += 1
        self.Tr_TPOS_Freq[(t, p)] += 1
        self.Tr_POS_Freq[p] += 1
        self.Tr_N_Term += 1

  def update_count2(self, t, p, val, bigrams=0):
    if bigrams == 0:
      if self.Term_Freq[t] + val >= 0:
        self.Term_Freq[t] += val
      if self.TPOS_Freq[(t, p)] + val >= 0:
        self.TPOS_Freq[(t, p)] += val
      if self.N_Term + val >= 0:
        self.N_Term += val
      if self.POS_Freq[p] + val >= 0:
        self.POS_Freq[p] += val
    elif bigrams == 1:
      if self.BN_Term + val >= 0:
        self.BN_Term += val
      if self.BTerm_Freq[t] + val >= 0:
        self.BTerm_Freq[t] += val
      if self.BPOS_Freq[p] + val >= 0:
        self.BPOS_Freq[p] += val
      if self.BTPOS_Freq[(t, p)] + val >= 0:
        self.BTPOS_Freq[(t, p)] += val

  def update_count3(self, t, p, vals, threshold, bigrams=0, type_d='in'):
    if bigrams == 0:
      self.Term_Freq[t] += threshold - vals['un_term_'+type_d]
      self.TPOS_Freq[(t, p)] += threshold - vals['un_tpos_'+type_d]
      self.POS_Freq[p] += threshold - vals['un_pos_'+type_d]
      self.N_Term += np.median([threshold - vals['un_term_'+type_d], threshold - vals['un_tpos_'+type_d], threshold - vals['un_pos_'+type_d]])
    elif bigrams == 1:
      self.BN_Term += np.median([threshold - vals['bi_term_'+type_d], threshold - vals['bi_pos_'+type_d], threshold - vals['bi_tpos_'+type_d]])
      self.BTerm_Freq[t] += threshold - vals['bi_term_'+type_d]
      self.BPOS_Freq[p] += threshold - vals['bi_pos_'+type_d]
      self.BTPOS_Freq[(t, p)] += threshold - vals['bi_tpos_'+type_d]

  def construct_features(self, sentences, use_smoothing=True):
    print 'creating features...'

    if not use_smoothing:
      self.set_lambda(0)

    data = []
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    for i, sent in enumerate(sentences):
      print i
      term, tpos, posf, bterm, btpos, bposf = (0, 0, 0, 0, 0, 0)
      tokenized_tagged = nltk.pos_tag(nltk.word_tokenize(sent))
      for token, p in tokenized_tagged:
        # unigrams
        try:
          new_token = regex.sub(u'', token).decode('utf-8')
          if not new_token == u'' and not new_token in stopwords.words('english'):
            term += self.Term_Freq[new_token]/self.N_Term
            # I think we need a different normalizer here
            posf += self.POS_Freq[p]/self.N_Term
            tpos += self.Term_Freq[(new_token, p)]/self.N_Term
        except:
          pass

      # normalize with respect to sentence length
      term /= len(sent)
      posf /= len(sent)
      tpos /= len(sent)

      # bigrams
      words = [elem[0] for elem in tokenized_tagged]
      pos_tags = [elem[1] for elem in tokenized_tagged]

      b_words = nltk.bigrams(words)
      b_pos = nltk.bigrams(pos_tags)

      if len(b_words) > 0:
        for b_w, b_p in zip(b_words, b_pos):
          bterm += self.BTerm_Freq[b_w]/self.BN_Term
          bposf += self.BPOS_Freq[b_p]/self.BN_Term
          btpos += self.BTPOS_Freq[(b_w, b_p)]/self.BN_Term

      # normalize
        bterm /= len(b_words)
        bposf /= len(b_pos)
        btpos /= len(b_words)

      data.append([term, posf, tpos, bterm, bposf, btpos])

    return np.asarray(data)

  def save(self, path='features_data.pickle'):
    self.log('save data to '+path)
    storage = {}

    # unigram
    storage['Term_Freq'] = self.Term_Freq
    storage['POS_Freq'] = self.POS_Freq
    storage['TPOS_Freq'] = self.TPOS_Freq
    storage['Trans_Freq'] = self.Trans_Freq
    storage['N_Term'] = self.N_Term
    storage['Tr_Term_Freq'] = self.Tr_Term_Freq
    storage['Tr_TPOS_Freq'] = self.Tr_TPOS_Freq
    storage['Tr_POS_Freq'] = self.Tr_POS_Freq
    storage['Tr_N_Term'] = self.Tr_N_Term
    # bigram
    storage['BPOS_Freq'] = self.BPOS_Freq
    storage['BTPOS_Freq'] = self.BTPOS_Freq
    storage['BTerm_Freq'] = self.BTerm_Freq
    storage['BN_Term'] = self.BN_Term

    with open(path, 'wb') as handle:
      pickle.dump(storage, handle)
    self.log('saved..')

  def load(self, path='features_data.pickle'):
    self.log('loading data '+path)
    with open(path, 'rb') as handle:
      storage = pickle.load(handle)

    # unigram
    self.Term_Freq = storage['Term_Freq']
    self.POS_Freq = storage['POS_Freq']
    self.TPOS_Freq = storage['TPOS_Freq']
    self.Trans_Freq = storage['Trans_Freq']
    self.N_Term = storage['N_Term']

    try :
    # if True:
      self.Tr_Term_Freq = storage['Tr_Term_Freq']
      self.Tr_TPOS_Freq = storage['Tr_TPOS_Freq']
      self.Tr_POS_Freq = storage['Tr_POS_Freq']
      self.Tr_N_Term = storage['Tr_N_Term']
    except :
      self.Tr_Term_Freq = defaultdict(int)
      self.Tr_TPOS_Freq = defaultdict(int)
      self.Tr_POS_Freq = defaultdict(int)
      self.Tr_N_Term = 0

    # bigram
    self.BPOS_Freq = storage['BPOS_Freq']
    self.BTPOS_Freq = storage['BTPOS_Freq']
    self.BTerm_Freq = storage['BTerm_Freq']
    self.BN_Term = storage['BN_Term']

    self.set_lambda(1)
    self.log('loaded..')

  def log(self, string):
    f = open('log_FE.txt', 'a+')
    print string
    f.write(string+'\n')
    f.close()

if __name__ == '__main__':
    F = Features()
    F.load('legal_in_model_noprune.pickle')
    b1= len(F.Term_Freq)
    # print F.Term_Freq.keys()
    b2= len(F.POS_Freq)
    b3= len(F.TPOS_Freq)

    b4= F.N_Term
    b5= len(F.BTerm_Freq)
    b6= len(F.BPOS_Freq)
    b7= len(F.BTPOS_Freq)
    b8= F.BN_Term
    F.parse_doc('project3_data_selection/legal.half.es',is_translation=True)
    # F.parse_doc('project3_data_selection/out.mixed.legal.es')
    F.save('legal_in_model_noprune_with_translation.pickle')
    # F.load('legal_in_model_noprune_with_translation.pickle')
    #F.save('sample_mix_model.pickle')
    # F.save('mix_model.pickle')
    # F.load('legal_in_model_es.pickle')
    print 'before'
    print b1,b2,b3,b4,b5,b6,b7,b8
    print len(F.Term_Freq)
    # print F.Term_Freq.keys()
    print len(F.POS_Freq)
    print len(F.TPOS_Freq)

    print F.N_Term
    print len(F.BTerm_Freq)
    print len(F.BPOS_Freq)
    print len(F.BTPOS_Freq)
    print F.BN_Term

    print len(F.Tr_Term_Freq)
    print len(F.Tr_TPOS_Freq)
    print len(F.Tr_POS_Freq)
    for i in range(100):
      print F.Tr_TPOS_Freq.keys()[i]

    print F.Tr_TPOS_Freq.keys().index(('dicha','aq0fsp'))

    print F.Tr_N_Term
    # print F.Term_Freq.keys()[0]#, F.Term_Freq(F.Term_Freq.keys()[0])
    # print F.POS_Freq[('even','RB')]#, F.POS_Freq(F.POS_Freq.keys()[0])
