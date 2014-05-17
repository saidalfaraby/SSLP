from __future__ import division
from collections import defaultdict
import numpy as np
import dill as pickle
import nltk

class IntelligentSelection(object):
  """docstring for IntelligentSelection"""
  def __init__(self, Features):
    self.Features = Features
    self.MixDoc = [] #store tuple of docID and pos tagged sentences


  def score(termpos1, termpos2=None):
  #score based on difference of cross entropy
  #sentences have been tokenized and pos tagged
    F = self.Features
    score1 = 0
    for t,p in termpos1:
      sore1 += -np.log(F.Term_Freq[t]/F.N_Term) + -np.log(F.POS_Freq[t,p]/F.N_Term)
    if termpos2==None:
      return score1/len(termpos1)
    score2 = 0
    for t,p in termpos2:
      sore2 += -np.log(F.Term_Freq[t]/F.N_Term) + -np.log(F.POS_Freq[t,p]/F.N_Term)
    return score1/len(termpos1) - score2/len(termpos2)

  def select(mix_doc, how_many=100, is_update=False):
    for mix in self.MixDoc:


  def parse_mix_doc(path):
    f = open(path)
    docId = 0
    for sentence in f.readlines():
      s = nltk.pos_tag(nltk.word_tokenize(sentence))
      self.MixDoc.append((docId, s))




    