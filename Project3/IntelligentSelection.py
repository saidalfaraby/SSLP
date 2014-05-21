from __future__ import division
from collections import defaultdict
import numpy as np
import dill as pickle
import nltk
from nltk.corpus import stopwords
from FeatureExtraction import Features
import re
import random
import string

class IntelligentSelection(object):
  """docstring for IntelligentSelection"""
  def __init__(self, In_Model, Mix_Model=None):
    self.In_Model = In_Model
    self.Mix_Model = Mix_Model
    self.Mix_Docs = [] #store object of docID and pos tagged sentences, and score
    self.Selected_Docs = []
    self.scores = []
    self.dual_score = True
    self.include_pos = False


  def entropy_score(self,termpos, find_threshold=False):
  #score based on difference of cross entropy
  #sentences have been tokenized and pos tagged
    
    IN = self.In_Model
    MIX = self.Mix_Model
    sc_term_in = 0
    sc_term_mix = 0
    sc_pos_in = 0
    sc_pos_mix = 0
    if len(termpos)==0 or MIX.N_Term<-99999:
      return -99999 if find_threshold else 99999
    for t,p in termpos:
      sc_term_in += -np.log(IN.Term_Freq[t]/IN.N_Term) 
      sc_pos_in+= -np.log(IN.POS_Freq[t,p]/IN.N_Term)
    if MIX!=None and self.dual_score:
      sc_term_mix += -np.log(MIX.Term_Freq[t]/MIX.N_Term) 
      sc_pos_mix+= -np.log(MIX.POS_Freq[t,p]/MIX.N_Term)
    if self.include_pos :
      return ((sc_term_in+sc_pos_in)-(sc_term_mix+sc_pos_mix))/len(termpos)
    return (sc_term_in-sc_term_mix)/len(termpos)
  def select(self, threshold, is_update=False):
    self.log('selecting ...')
    for mix in self.Mix_Docs:
      mix['score'] = self.entropy_score(mix['termpos'])
      if mix['docID']<450000 and mix['score']<99999 and mix['score']>-99999:
        self.scores.append(mix['score'])
      if mix['score'] > threshold:
        self.Selected_Docs.append(mix['docID'])
        
    self.log('finish selecting')

  def measure(self,label):
    self.log('in domain : '+str(len(label)))
    self.log('retrieve :'+str(len(self.Selected_Docs)))
    intersection = list(set(label)&set(self.Selected_Docs))
    self.log('intersection : '+str(len(intersection)))
    return {'precision': len(intersection)/len(self.Selected_Docs), 'recall':len(intersection)/len(label)}

  def findThreshold(self):
    self.log('finding threshold')
    maxScore = -99999999
    allScore = []
    for mix in self.Mix_Docs:
      score = self.entropy_score(mix['termpos'],True)
      if score > -99999:
        allScore.append(score)
      if score>maxScore:
        maxScore = score
    self.log('threshold statistics')
    self.log('threshold = '+str(maxScore))
    allScore = np.asarray(allScore)
    self.log('max score '+str(np.max(allScore)))
    self.log('min score '+str(np.min(allScore)))
    self.log('avg score '+str(np.mean(allScore)))
    self.log('std score '+str(np.std(allScore)))
    return maxScore

  def parse_mix_doc(self,path):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    self.log('parsing documents from '+path)
    self.Mix_Docs = []
    f = open(path)
    docId = 0
    for sentence in f.readlines():
      sentence = nltk.pos_tag(nltk.word_tokenize(sentence))
      s = []
      for token,pos in sentence: 
        try :
          new_token = regex.sub(u'', token).decode('utf-8')
          if not new_token == u'' and not new_token in stopwords.words('english'):
            s.append((new_token,pos))
        except :
          pass
      self.Mix_Docs.append({'docID':docId, 'termpos':s, 'score':-999999})
      docId +=1
    self.log('parsed..')

  def stats(self):
    self.log('length '+str(len(self.scores)))
    n = np.asarray(self.scores)
    self.log('min '+str(np.min(n)))
    self.log('max '+str(np.max(n)))
    self.log('mean '+str(np.mean(n)))
    self.log('std '+str(np.std(n)))

  def save(self, path='mix_doc.pickle'):
    self.log('save data to '+path)
    with open(path, 'wb') as handle:
      pickle.dump(self.Mix_Docs, handle)
    self.log('saved..')

  def load(self, path='mix_doc.pickle'):
    self.log('loading data '+path)
    with open(path, 'rb') as handle:
      self.Mix_Docs = pickle.load(handle)
    self.log('loaded..')
    self.log('number of docs loaded : '+str(len(self.Mix_Docs)))

  def log(self,string):
    f = open('log_IS.txt', 'a+')
    print string
    f.write(string+'\n')
    f.close()

if __name__ == '__main__':
  th = 10

  In_Model = Features()
  Mix_Model = Features()
  In_Model.load()
  IS = IntelligentSelection(In_Model,Mix_Model)
  IS.include_pos = False
  IS.dual_score = True
  IS.log('---------------------------')
  IS.log('run with threshold = '+str(th))
  # IS.parse_mix_doc('project3_data_selection/legal.dev.en')
  # th = IS.findThreshold()
  # IS.parse_mix_doc('project3_data_selection/out.mixed.legal.en')
  # IS.save('legal_dev_doc.pickle')
  IS.load()
  
  for idx in random.sample(range(500000),50000):
    for t,p in IS.Mix_Docs[idx]['termpos']:
      Mix_Model.update_count(t,p)
  print Mix_Model.N_Term
  print len(Mix_Model.Term_Freq)
  IS.Mix_Model = Mix_Model
  # Mix_Model.save('mix_model.pickle')
  # Mix_Model.load('mix_model.pickle')
  

  # IS.parse_mix_doc('project3_data_selection/legal.dev.en')
  # th = IS.findThreshold()
  # IS.parse_mix_doc('project3_data_selection/out.mixed.legal.en')
  # IS.save('legal_dev_doc.pickle')

  # th=IS.findThreshold()
  IS.select(th)
  # label = range(2000)
  label = range(450000,500000)
  m = IS.measure(label)
  IS.log('precision '+str(m['precision']))
  IS.log('recall '+str(m['recall']))

  IS.stats()
  IS.log('\n\n')





    