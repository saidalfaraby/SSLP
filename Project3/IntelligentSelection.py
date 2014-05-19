from __future__ import division
from collections import defaultdict
import numpy as np
import dill as pickle
import nltk
from nltk.corpus import stopwords
from FeatureExtraction import Features
import re
import string


class IntelligentSelection(object):
  """docstring for IntelligentSelection"""
  def __init__(self, Features):
    self.Features = Features
    self.MixDoc = []  # store object of docID and pos tagged sentences, and score
    self.selectedDoc = []
    self.scores = []

  def entropy_score(self, termpos1, termpos2=None):
  #score based on difference of cross entropy
  #sentences have been tokenized and pos tagged

    F = self.Features
    score1 = 0
    score2 = 0
    for t, p in termpos1:
      score1 += -np.log(F.Term_Freq[t]/F.N_Term) + -np.log(F.POS_Freq[t, p]/F.N_Term)
    try:
      score1 /= len(termpos1)
    except:
      score1 = 99999
    if termpos2 is not None:
      for t, p in termpos2:
        score2 += -np.log(F.Term_Freq[t]/F.N_Term) + -np.log(F.POS_Freq[t, p]/F.N_Term)
      try:
        score2 /= len(termpos2)
      except:
        score2 = 99999
    return score1 - score2

  def select(self, threshold, is_update=False):
    self.log('selecting ...')
    for mix in self.MixDoc:
      mix['score'] = self.entropy_score(mix['termpos'])
      if mix['score'] < 99999:
        self.scores.append(mix['score'])
      if mix['score'] < threshold:
        self.selectedDoc.append(mix['docID'])

    self.log('finish selecting')

  def measure(self, label):
    print 'label', len(label)
    print 'retrieve', len(self.selectedDoc)
    intersection = list(set(label) & set(self.selectedDoc))
    print 'inter', len(intersection)
    return {'precision': len(intersection)/len(self.selectedDoc), 'recall': len(intersection)/len(label)}

  def findThreshold(self):
    self.log('finding threshold')
    maxScore = -99999999
    for mix in self.MixDoc:
      score = self.entropy_score(mix['termpos'])
      if score > maxScore:
        maxScore = score
    self.log('threshold = '+str(maxScore))
    return maxScore

  def parse_mix_doc(self, path):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    self.log('parsing documents from '+path)
    self.MixDoc = []
    f = open(path)
    docId = 0
    for sentence in f.readlines():
      sentence = nltk.pos_tag(nltk.word_tokenize(sentence))
      s = []
      for token, pos in sentence:
        try:
          new_token = regex.sub(u'', token).decode('utf-8')
          if not new_token == u'' and not new_token in stopwords.words('english'):
            s.append((new_token, pos))
        except:
          pass
      self.MixDoc.append({'docID': docId, 'termpos': s, 'score': -999999})
      docId += 1
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
      pickle.dump(self.MixDoc, handle)
    self.log('saved..')

  def load(self, path='mix_doc.pickle'):
    self.log('loading data '+path)
    with open(path, 'rb') as handle:
      self.MixDoc = pickle.load(handle)
    self.log('loaded..')

  def log(self, string):
    f = open('log_IS.txt', 'a+')
    print string
    f.write(string+'\n')
    f.close()

if __name__ == '__main__':
  th = 17.3

  Model = Features()
  Model.load()
  IS = IntelligentSelection(Model)
  IS.log('---------------------------')
  IS.log('run with threshold = '+str(th))
  # IS.parse_mix_doc('project3_data_selection/legal.dev.en')
  # th = IS.findThreshold()
  # IS.parse_mix_doc('project3_data_selection/out.mixed.legal.en')
  # IS.save()
  # IS.load()
  IS.select(th)
  # label = range(2000)
  label = range(450000, 500000)
  m = IS.measure(label)
  IS.log('precision '+str(m['precision']))
  IS.log('recall '+str(m['recall']))

  IS.stats()
  IS.log('\n\n')
