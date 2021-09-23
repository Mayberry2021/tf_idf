import math
import copy
import loader
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from konlpy.tag import Mecab, Okt
from preprocess import Cleaning_Noise
from tensorflow.keras.preprocessing.text import Tokenizer

class TF_IDF_FUNCTION(object):
	def __init__(self):
		self.okt = Okt()
		self.tokenizer = Tokenizer()
		self.stopwords_pos = ['Determiner','Adverb','Conjunction','Exclamation','PreEomi','Eomi','Suffix']		
		self.stopwords_set = ['은','는','이','가','을','를','으로','로','하다','되다']
		self.base_dtm = {}

	def get_word_index(self):
		return self.base_dtm.keys()

	def document_sum(self, document):
		dsum = ''
		for docu in document:
			dsum += docu
		return dsum

	def dtm2(self, document):
		dsum = self.document_sum(document)
		tokenized = self.okt.pos(dsum)
		result = [word for word, tag in tokenized if tag not in self.stopwords_pos if len(word) > 1]
		self.tokenizer.fit_on_texts(result)
		self.base_dtm = {v:0 for v,_ in self.tokenizer.word_counts.items()}
		dtm_matrix = []
		for docu in document:
			temp_base_dtm = copy.deepcopy(self.base_dtm)
			d_tokenized = self.okt.morphs(docu)
			for word in d_tokenized:
				if word in self.base_dtm.keys():
					temp_base_dtm[word] += 1
			else:
				dtm_matrix.append(list(temp_base_dtm.values()))
		return dtm_matrix


	def dtm(self, document):
		dsum = self.document_sum(document)
		tokenized = self.okt.morphs(dsum)
		result = [word for word in tokenized if word not in self.stopwords_set if len(word) > 1]
		self.tokenizer.fit_on_texts(result)
		self.base_dtm = {v:0 for v,_ in self.tokenizer.word_counts.items()}
		dtm_matrix = []
		for docu in document:
			temp_base_dtm = copy.deepcopy(self.base_dtm)
			d_tokenized = self.okt.morphs(docu)
			for word in d_tokenized:
				if word in self.base_dtm.keys():
					temp_base_dtm[word] += 1
			else:
				dtm_matrix.append(list(temp_base_dtm.values()))
		return dtm_matrix

	def idf(self, word_index:int, dtm_matrix:list, docu_count:int): # dtm_matrix -> 2d list
		df = 0
		for dtm in dtm_matrix:
			if (dtm[word_index]):
				df += 1
		return math.log(docu_count / (1+df))

	def tf_idf(self, dtm_matrix:list):
		tf_idf_matrix = []
		for dtm in dtm_matrix:
			tf_idf_value_set = []
			for idx in range(0, len(dtm)):
				tf = dtm[idx]
				idf = self.idf(idx, dtm_matrix, len(dtm_matrix))
				value = tf*idf
				tf_idf_value_set.append(value)
			else:
				tf_idf_matrix.append(tf_idf_value_set)
		return tf_idf_matrix

	def cos_sim(self, matrix1:list, matrix2:list):
		return dot(matrix1, matrix2) / (norm(matrix1)*norm(matrix2))

	def docu_sim(self, tf_idf_matrix):
		num_set = list(range(len(tf_idf_matrix)))
		for a in num_set:
			if a == num_set[-1]: break
			for b in num_set[a+1:]:
				sim_value = self.cos_sim(tf_idf_matrix[a], tf_idf_matrix[b])
				print(f'문서{a+1}와 문서{b+1}의 유사도는 {sim_value}입니다.')
			

# use_sample
'''
a = ['먹고 싶은 사과', '먹고 싶은 바나나', '길고 노란 바나나 바나나', '저는 과일이 좋아요']
c = TF_IDF_FUNCTION()
mtx = c.dtm(a)
tf_idf_matrix = c.tf_idf(mtx)
c.docu_sim(tf_idf_matrix)
'''
