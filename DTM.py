from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import *
from db_controller import *
from konlpy.tag import Okt, Kkma, Mecab
from numpy.linalg import norm
from numpy import dot
import numpy as np
import os
import sys

def text_slice(documents:list): # db에서 꺼낸 기사 데이터 정제 -> [' 기사본문 ', ' 기사본문 ', ... ' 기사본문 ']의 형태
	cn = Cleaning_Noise()
	result = []
	for docu in documents:
		clean_data = cn.cleaning(docu[0].strip('\n'))
		result.append(clean_data)	
	return result

def train_set(db_names:list, section:str):
	train_set = []
	titles = select_title(db_names, section)
	train_set += titles 
	return train_set

def tf_idf(data:list):
	tf_idfv = TfidfVectorizer().fit(data)
	tf_matrix = tf_idfv.transform(data).toarray()	
	return tf_matrix

def cos_sim(matrix:list):
	return dot(matrix[0], matrix[1]) / (norm(matrix[0])*norm(matrix[1]))

def most_similar(titles, input_data:str):
	with open('new_lab.txt', 'a', encoding='utf-8') as file:
		new = []
		new += titles
		new.append(input_data)
		tf_idfv = TfidfVectorizer().fit(new)
		tf_matrix = tf_idfv.transform(new).toarray()
		num_set = list(range(len(tf_matrix)-1))
		max, max_idx = 0, 0
		for idx in num_set:
			data = [tf_matrix[-1], tf_matrix[idx]]
			value = cos_sim(data)
			if value > max:
				max = value
				max_idx = idx
			else:
				pass
		file.write('입력값:' + '\t' + input_data + '\t' + '기존값:' + titles[max_idx] + '의 유사도' + ':' + str(max) + '\n')
		new.remove(input_data)