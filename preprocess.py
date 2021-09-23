from konlpy.tag import Okt, Kkma
import numpy as np
import copy
import re

class Cleaning_Noise(object):
	def __init__(self):
		pass

	def cleaning(self, corpus):
		result = re.sub('["0-9a-zA-Zㄱ-ㅎㅏ-ㅣ~!@#$%~&-_=\\n\.\'\$\(\)\*\+\?\[\\\^\{\`♥␞]', '', corpus)
		return result

	def clean_noise_file(self, read_file, write_file):
		with open(f'{read_file}','r',encoding='utf-8') as file, open(f'{write_file}', 'a', encoding='utf-8') as file2:
			while True:
				raw = file.readline()
				if (not raw):
					break
				result = self.cleaning(raw)
				file2.write(result)

	def clean_noise_data(self, tokenized_data):
		result = []
		for sentence in tokenized_data:
			result.append(self.cleaning(sentence))
		return result