"""
utils functions to preprocess the input file
"""

import nltk
import re 
import numpy as np

def remove_first_useless_par(line):	
	tmp = line.split(' ')
	tmp[-1] = tmp[-1][: -1]
	return ' '.join(tmp[1:])

def remove_useless_tag(line):
	tmp = line.split(' ')
	for i, w in enumerate(tmp):
		if w[0] == '(':
			tmp[i] = re.sub(r'-.+', '', w)
	return ' '.join(tmp)


def preprocess(lines):
	preprocessed = []
	for line in lines:
		preprocessed.append(remove_useless_tag(remove_first_useless_par(line)))
	return preprocessed

