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
import bisect
import sys

class IntelligentSelection(object):
  """docstring for IntelligentSelection"""
  def __init__(self, In_Model, Mix_Model=None):
    self.In_Model = In_Model
    self.Mix_Model = Mix_Model
    self.Mix_Docs = [] #store object of docID and pos tagged sentences, and score
    self.Mix_Docs_Translation = []
    self.Selected_Docs = []
    self.scores = []
    self.dual_score = True
    self.include_pos = False
    self.is_unigram = True
    self.is_bigram = True
    self.is_translation = False
    self.is_length_normalized = False
    self.log_filename = 'log_IS.txt'
    self.type_score = ['un_term_in', 'bi_term_in', 'un_pos_in', 'bi_pos_in', 'un_tpos_in', 'bi_tpos_in',
                  'un_term_mix', 'bi_term_mix', 'un_pos_mix', 'bi_pos_mix', 'un_tpos_mix', 'bi_tpos_mix']


  def entropy_score(self,termpos, tr_termpos=[], find_threshold=False):
  #score based on difference of cross entropy
  #sentences have been tokenized and pos tagged
    IN = self.In_Model
    MIX = self.Mix_Model
    score_per_each = {}
    sc_term_in, sc_term_mix, sc_pos_in,\
     sc_pos_mix, sc_pos_in_only, sc_pos_mix_only, \
     sc_tr_term_in, sc_tr_term_mix = (0,0,0,0,0,0,0,0)
    sc_bi_term_in, sc_bi_pos_in, sc_bi_term_mix, \
      sc_bi_pos_mix, sc_bi_pos_in_only, sc_bi_pos_mix_only = (0,0,0,0,0,0)

    if len(termpos)==0 or MIX.N_Term<-99999:
      for i, score in enumerate(self.type_score):
        score_per_each[score] = 0
      if find_threshold:
        return -99999, score_per_each
      return 99999, score_per_each

    sent_length = 1
    if self.is_length_normalized :
      sent_length = len(termpos)

    if self.is_unigram:
      for t,p in termpos:
        pt = IN.Term_Freq[t]/IN.N_Term
        sc_term_in += -np.log(pt)
        ptp = IN.TPOS_Freq[(t,p)]/IN.N_Term
        sc_pos_in+= -np.log(ptp)
        pp = IN.POS_Freq[p]/IN.N_Term
        sc_pos_in_only += -np.log(pp)
        if MIX!=None and self.dual_score:
          mpt = MIX.Term_Freq[t]/MIX.N_Term
          sc_term_mix += -np.log(mpt)
          mptp = MIX.TPOS_Freq[(t,p)]/MIX.N_Term
          sc_pos_mix+= -np.log(mptp)
          mpp = MIX.POS_Freq[p]/MIX.N_Term
          sc_pos_mix_only += -np.log(mpp)

    if self.is_translation:
      for t,p in tr_termpos:
        sc_tr_term_in += -np.log(IN.Tr_Term_Freq[t]/IN.Tr_N_Term)
        if MIX!=None and self.dual_score:
          sc_tr_term_mix += -np.log(MIX.Tr_Term_Freq[t]/MIX.Tr_N_Term)  


    if self.is_bigram:
      # bigrams
      b_words, b_pos = self.In_Model.to_bigram(termpos)
      # words = [elem[0] for elem in termpos]
      # pos_tags = [elem[1] for elem in termpos]
      # b_words = nltk.bigrams(words)
      # b_pos = nltk.bigrams(pos_tags)
      if len(b_words) > 0:
        for t, p in zip(b_words, b_pos):
          sc_bi_term_in += -np.log(IN.BTerm_Freq[t]/IN.BN_Term)
          sc_bi_pos_in += -np.log(IN.BTPOS_Freq[(t, p)]/IN.BN_Term)
          sc_bi_pos_in_only += -np.log(IN.BPOS_Freq[p]/IN.BN_Term)
          if MIX!=None and self.dual_score:
            sc_bi_term_mix += -np.log(MIX.BTerm_Freq[t]/MIX.BN_Term)
            sc_bi_pos_mix+= -np.log(MIX.BTPOS_Freq[(t,p)]/MIX.BN_Term)
            sc_bi_pos_mix_only += -np.log(MIX.BPOS_Freq[p]/MIX.BN_Term)

      final_scores = [sc_term_in, sc_bi_term_in, sc_pos_in_only, sc_bi_pos_in_only, sc_pos_in, sc_bi_pos_in,
                    sc_term_mix, sc_bi_term_mix, sc_pos_mix_only, sc_bi_pos_mix_only, sc_pos_mix, sc_bi_pos_mix]
      for i, score in enumerate(self.type_score):
        if i < 6:
          score_per_each[score] = final_scores[i] - final_scores[i+6]
        else:
          score_per_each[score] = final_scores[i] - final_scores[i-6]

      sc_term_in += sc_bi_term_in
      sc_pos_in += sc_bi_pos_in
      sc_pos_in_only += sc_bi_pos_in_only
      sc_term_mix += sc_bi_term_mix
      sc_pos_mix += sc_bi_pos_mix
      sc_pos_mix_only += sc_bi_pos_mix_only
    if self.include_pos and self.include_pos_only:
      # return ((sc_term_in+sc_pos_in)-(sc_term_mix+sc_pos_mix))/len(termpos)
      return ((sc_term_in+sc_pos_in+sc_pos_in_only+sc_tr_term_in)-(sc_term_mix+sc_pos_mix+sc_pos_mix_only+sc_tr_term_mix))/sent_length, score_per_each
    elif self.include_pos:
      return ((sc_term_in+sc_pos_in+sc_tr_term_in)-(sc_term_mix+sc_pos_mix+sc_tr_term_mix))/sent_length, score_per_each
    # return (sc_term_in-sc_term_mix)/len(termpos)
    return ((sc_term_in+sc_tr_term_in)-(sc_term_mix+sc_tr_term_mix))/sent_length, score_per_each

  def ratio_score(self,termpos, find_threshold=False, is_bigram = True):
  #score based on difference of cross entropy
  #sentences have been tokenized and pos tagged

    IN = self.In_Model
    MIX = self.Mix_Model
    sc_term_in, sc_term_mix, sc_pos_in, sc_pos_mix = (1,1,1,1)
    sc_bi_term_in, sc_bi_pos_in, sc_bi_term_mix, sc_bi_pos_mix = (1,1,1,1)

    if len(termpos)==0 or MIX.N_Term<-99999:
      return 0 if find_threshold else 0

    if self.is_unigram:
      for t,p in termpos:
        sc_term_in *= IN.Term_Freq[t]/IN.N_Term
        sc_pos_in*= IN.TPOS_Freq[t,p]/IN.N_Term
        if MIX!=None and self.dual_score:
          sc_term_mix *= MIX.Term_Freq[t]/MIX.N_Term
          sc_pos_mix*= MIX.TPOS_Freq[t,p]/MIX.N_Term

    if self.is_bigram:
      b_words, b_pos = self.In_Model.to_bigram(termpos)
      if len(b_words) > 0:
        for b_w, b_p in zip(b_words, b_pos):
          sc_bi_term_in *= IN.BTerm_Freq[t]/IN.BN_Term
          sc_bi_pos_in*= IN.BTPOS_Freq[t,p]/IN.BN_Term
          if MIX!=None and self.dual_score:
            sc_bi_term_mix *= MIX.BTerm_Freq[t]/MIX.BN_Term
            sc_bi_pos_mix*= MIX.BTPOS_Freq[t,p]/MIX.BN_Term

      sc_term_in *= sc_bi_term_in
      sc_pos_in *= sc_bi_pos_in
      sc_term_mix *= sc_bi_term_mix
      sc_pos_mix *= sc_bi_pos_mix
      # if IN.N_Term==0 or MIX.N_Term ==0 or MIX.Term_Freq[t]==0 or MIX.POS_Freq[t,p]==0:
      #   print 'something is zero'
      #   print 'TERM ',MIX.Term_Freq[t]
      #   print 'POS', MIX.POS_Freq[t,p]
    if self.include_pos :
      # return ((sc_term_in+sc_pos_in)-(sc_term_mix+sc_pos_mix))/len(termpos)
      try :
        return (sc_term_in*sc_pos_in)/(sc_term_mix*sc_pos_mix)
      except :
        return 99999
    # return (sc_term_in-sc_term_mix)/len(termpos)
    try :
      return sc_term_in/sc_term_mix
    except :
      return 99999

  def select(self, threshold=None, is_update=False, retrieve_per_iteration=None, n_iteration=1):

    self.log('selecting documents with threshold '+str(threshold)+' ...')
    self.log('dual score : '+str(self.dual_score))
    self.log('include pos : '+str(self.include_pos))
    self.log('include pos only:' + str(self.include_pos_only))
    self.log('unigram : '+str(self.is_unigram))
    self.log('bigram : '+str(self.is_bigram))
    self.log('is translation : '+str(self.is_translation))
    self.log('is update : '+str(is_update))
    self.log('retrieve per iteration : '+str(retrieve_per_iteration))
    self.log('number of iteration : '+str(n_iteration))
    self.log('is length normalized : '+str(self.is_length_normalized))
    label = range(450000,500000)
    scoring_function = self.entropy_score
    if threshold!=None and not is_update:
      if not self.is_translation :
        for mix in self.Mix_Docs:
          mix['score'], smt = scoring_function(mix['termpos'])
          if mix['docID']<450000 and mix['score']<99999 and mix['score']>-99999:
            self.scores.append(mix['score'])
          if mix['score'] < threshold:
            self.Selected_Docs.append(mix['docID'])
      else :
        for i in range(len(self.Mix_Docs)):
          mix = self.Mix_Docs[i]
          try :
            tr = self.Mix_Docs_Translation[i]
          except :
            print i
          mix['score'], smt = scoring_function(mix['termpos'],tr['termpos'])
          if mix['docID']<450000 and mix['score']<99999 and mix['score']>-99999:
            self.scores.append(mix['score'])
          if mix['score'] < threshold:
            self.Selected_Docs.append(mix['docID'])
    if threshold !=None and is_update:
      it = 0
      lr = 1
      decay = 0.8
      selected_documents = []
      total_docs = []
      for i in range(n_iteration):
        unselected_docs = []
        it+=1
        random.shuffle(self.Mix_Docs)
        print 'iteration : ', it
        for mix in self.Mix_Docs:
          #print self.entropy_score(mix['termpos'])
          mix['score'], mix['score_each'] = self.entropy_score(mix['termpos'])
          if mix['score'] < threshold:
            self.Selected_Docs.append(mix['docID'])
            selected_documents.append(mix)
          else:
            unselected_docs.append(mix)

        for mix in selected_documents:
          #update unigram model
          for token, pos in mix['termpos']:
            # self.In_Model.update_count2(token,pos, lr*(1 + threshold-mix['score']))
            # self.Mix_Model.update_count2(token, pos, -lr * np.exp(threshold-mix['score']))
            self.In_Model.update_count3(token, pos, mix['score_each'], threshold)
            # self.Mix_Model.update_count3(token, pos, mix['score_each'])
            #update bigram model
          b_words, b_pos = self.In_Model.to_bigram(mix['termpos'])
          for b_w, b_p in zip(b_words, b_pos):
            # self.In_Model.update_count2(b_w, b_p, lr*( 1 + threshold-mix['score']), bigrams=1)
            # self.Mix_Model.update_count2(b_w, b_p, -lr * np.exp(threshold-mix['score']), bigrams=1)
            self.In_Model.update_count3(b_w, b_p, mix['score_each'], threshold, bigrams=1)
        self.measure(label)
        self.Selected_Docs = []
        total_docs += selected_documents
        if len(selected_documents) == 0:
          print 'Converged'
          break
        selected_documents = []
        lr *= decay
        self.Mix_Docs = unselected_docs
      self.Selected_Docs = [mix['docID'] for mix in total_docs]

    elif is_update and retrieve_per_iteration!=None:
      it = 0
      for i in range(n_iteration):
        it+=1
        print 'iteration : ',it
        for mix in self.Mix_Docs:
          mix['score'] = self.entropy_score(mix['termpos'])
          # if mix['docID']<450000 and mix['score']<99999 and mix['score']>-99999:
            # self.scores.append(mix['score'])
        sorted_Mix_Docs = sorted(self.Mix_Docs, key=lambda k: k['score'])
        # self.Selected_Docs.extend(sorted_Mix_Docs[0:retrieve_per_iteration])
        for j in range(retrieve_per_iteration):
          doc = sorted_Mix_Docs[j]
          self.Selected_Docs.append(doc['docID'])
          #update unigram model
          for token, pos in doc['termpos']:
            self.In_Model.update_count(token,pos)
          #update bigram model
          b_words, b_pos = self.In_Model.to_bigram(doc['termpos'])
          for b_w, b_p in zip(b_words, b_pos):
            self.In_Model.update_count(b_w, b_p, bigrams=1)
        self.Mix_Docs = sorted_Mix_Docs[retrieve_per_iteration:]


    self.log('finish selecting')

  def measure(self,label):
    self.log('in domain : '+str(len(label)))
    self.log('retrieve :'+str(len(self.Selected_Docs)))
    intersection = list(set(label)&set(self.Selected_Docs))
    self.log('intersection : '+str(len(intersection)))
    try :
      return {'precision': len(intersection)/len(self.Selected_Docs), 'recall':len(intersection)/len(label)}
    except :
      return {'precision': 0, 'recall':len(intersection)/len(label)}

  def findThreshold(self):
    self.log('finding threshold')
    maxScore = -99999999
    allScore = []
    for mix in self.Mix_Docs:
      score, smt = self.entropy_score(mix['termpos'],find_threshold = True)
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
    self.log('median score '+str(np.median(allScore)))
    self.log('std score '+str(np.std(allScore)))
    return maxScore

  def parse_mix_doc(self,path):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    self.log('parsing documents from '+path)
    self.Mix_Docs = []
    f = open(path)
    docId = 0
    if not self.is_translation :
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
    else :
      spanish_tagger = pickle.load(open('bitag_spanish.pickle','rb'))
      for sentence in f.readlines():
        term_pos = spanish_tagger.tag(nltk.word_tokenize(sentence))
        s = []
        for term,pos in term_pos :
          if not term == '' and not term in stopwords.words('spanish'):
            s.append((term,pos))
        self.Mix_Docs_Translation.append({'docID':docId, 'termpos':s, 'score':-999999})
        docId +=1

    self.log('parsed..')

  def stats(self):
    self.log('length '+str(len(self.scores)))
    n = np.asarray(self.scores)
    self.log('min '+str(np.min(n)))
    self.log('max '+str(np.max(n)))
    self.log('mean '+str(np.mean(n)))
    self.log('median '+str(np.median(n)))
    self.log('std '+str(np.std(n)))

  def save(self, path=None, translation_path=None):
    if path != None :
      self.log('save mix doc data to '+path)
      with open(path, 'wb') as handle:
        pickle.dump(self.Mix_Docs, handle)
    if translation_path!=None:
      self.log('save trans mix doc data to '+path)
      with open(translation_path, 'wb') as handle:
        pickle.dump(self.Mix_Docs_Translation, handle)
    self.log('saved..')

  def load(self, path=None, translation_path=None):
    if path!=None:
      self.log('loading data '+path)
      with open(path, 'rb') as handle:
        self.Mix_Docs = pickle.load(handle)
      self.log('number of docs loaded : '+str(len(self.Mix_Docs)))
    if translation_path !=None and self.is_translation:
      self.log('loading translation data '+translation_path)
      with open(translation_path, 'rb') as handle:
        self.Mix_Docs_Translation = pickle.load(handle)
        self.log('number of translation docs loaded : '+str(len(self.Mix_Docs_Translation)))
    self.log('loaded..')


  def log(self,string):
    f = open(self.log_filename, 'a+')
    print string
    f.write(string+'\n')
    f.close()

  def write_doc(self, in_filename, out_filename):
    self.log('writing selected docs from '+in_filename)
    self.log('to '+out_filename)
    fo = open(out_filename, 'wb')
    fi = open(in_filename,'rb')
    doc = fi.readlines()
    for i in self.Selected_Docs:
      fo.write(doc[i]+'\n')
    fo.close()
    fi.close()

if __name__ == '__main__':
  
  in_model_file = 'legal_in_model_noprune.pickle'
  mix_model_file = 'legal_mix_model_noprune.pickle'
  In_Model = Features()
  Mix_Model = Features()
  In_Model.load(in_model_file)
  Mix_Model.load(mix_model_file)
  print In_Model.N_Term
  print In_Model.Tr_N_Term
  In_Model.set_lambda(0.1)
  Mix_Model.set_lambda(0.1)
  IS = IntelligentSelection(In_Model,Mix_Model)
  IS.log_filename = 'log1.txt'
  if len(sys.argv) > 1:
    IS.log_filename = sys.argv[1]
  IS.log('\n\n---------------------------')
  IS.log('in domain LM file : '+in_model_file)
  IS.log('mix domain LM file : '+mix_model_file)
  IS.include_pos = True
  IS.include_pos_only = False
  IS.dual_score = True
  IS.is_unigram = True
  IS.is_bigram = False
  IS.is_translation = False
  IS.is_length_normalized = True
  

  # IS.log('run with threshold = '+str(th))
  # IS.parse_mix_doc('project3_data_selection/legal.dev.en')
  # th = IS.findThreshold()
  # IS.parse_mix_doc('project3_data_selection/out.mixed.software.en')
  # IS.save('software_mix_doc.pickle')
  # IS.load('legal_dev_doc.pickle')
  # th = IS.findThreshold()
  # for idx in random.sample(range(500000),50000):
  #   for t,p in IS.Mix_Docs[idx]['termpos']:
  #     Mix_Model.update_count(t,p)
  # print Mix_Model.N_Term
  # print len(Mix_Model.Term_Freq)
  # Mix_Model.set_lambda(1)
  # IS.Mix_Model = Mix_Model

  # Mix_Model.save('mix_model.pickle')
  # Mix_Model.load('mix_model.pickle')


  # IS.parse_mix_doc('project3_data_selection/legal.dev.en')
  # th = IS.findThreshold()
  # IS.parse_mix_doc('project3_data_selection/out.mixed.legal.es')
  # IS.save(translation_path='mix_doc_translation.pickle')

  # th=IS.findThreshold()
  # IS.load('software_mix_doc.pickle','mix_doc_translation.pickle')
  IS.load('software_mix_doc.pickle')
  # IS.load('software_mix_doc.pickle')
  # IS.select(threshold = -20, is_update=True, retrieve_per_iteration=None, n_iteration=5)
  IS.select(0.0)
  # label = range(2000)
  label = range(450000,500000)
  m = IS.measure(label)
  IS.log('precision '+str(m['precision']))
  IS.log('recall '+str(m['recall']))
  IS.log('F1 '+str((2*m['precision']*m['recall'])/(m['precision']+m['recall'])))
  IS.write_doc('project3_data_selection/out.mixed.legal.en', 'selected.en')
  IS.write_doc('project3_data_selection/out.mixed.legal.es', 'selected.es')
  # IS.stats()
  IS.log('\n\n')
