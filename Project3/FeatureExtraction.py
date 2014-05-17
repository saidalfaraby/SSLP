from __future__ import division
from collections import defaultdict
import nltk
import numpy as np
import re
import string
from nltk.corpus import stopwords
import dill as pickle

class Features(object):
  """docstring for Features"""
  def __init__(self):
    self.Term_Freq = defaultdict(lambda:1) #term frequency. Key = term, Val = frequency
    self.POS_Freq = defaultdict(lambda:1) #postag frequency. Key = (term,POSTag), Val = Frequency
    self.Trans_Freq = defaultdict(lambda:1) #translation frequency. Key = (E,F), Val = frequency
    self.N_Term = 0

  def parse_doc(self, path):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    f = open(path)
    i=0
    for sentence in f.readlines():
      sentence = nltk.pos_tag(nltk.word_tokenize(sentence))
      i+=1
      print i
      #remove punctuation
      new_s = []
      for token,pos in sentence: 
        try :
          new_token = regex.sub(u'', token).decode('utf-8')
          if not new_token == u'' and not new_token in stopwords.words('english'):
            self.update_count(new_token,pos)
        except :
          pass
          
  
  def update_count(self,t,p):
    self.Term_Freq[t]+=1
    self.POS_Freq[(t,p)]+=1
    self.N_Term += 1

  def save(self, path='features_data.pickle'):
    storage = {}
    storage['Term_Freq'] = self.Term_Freq
    storage['POS_Freq'] = self.POS_Freq
    storage['Trans_Freq'] = self.Trans_Freq
    storage['N_Term'] = self.N_Term
    with open(path, 'wb') as handle:
      pickle.dump(storage, handle)

  def load(self, path='features_data.pickle'):
    with open(path, 'rb') as handle:
      storage = pickle.load(handle)
    self.Term_Freq = storage['Term_Freq']
    self.POS_Freq = storage['POS_Freq']
    self.Trans_Freq = storage['Trans_Freq']
    self.N_Term = storage['N_Term']


if __name__ == '__main__':
    F = Features()
    F.parse_doc('project3_data_selection/test')
    # F.save()
    # F.load()
    print len(F.Term_Freq)
    print len(F.POS_Freq)
    print F.N_Term
    print F.Term_Freq.keys()[0]#, F.Term_Freq(F.Term_Freq.keys()[0])
    print F.POS_Freq[('even','RB')]#, F.POS_Freq(F.POS_Freq.keys()[0])




