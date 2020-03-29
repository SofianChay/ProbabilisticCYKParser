"""
evaluation file 
"""
import numpy as np
import re 

from preprocessing import preprocess 
from extract_pcfg import pcfg
from cyk_algorithm import probabilistic_parser


ponct = ['.', '!', '?', '...', ':', ';', ',']

def tokenize(line):
	tokens = []
	tmp = line.split(' ')
	period_added = False
	for elt in tmp:
		if elt[-1] == ')':
			word = ''
			j = 0
			while elt[j] != ')':
				word += elt[j]
				j += 1
			tokens.append(word)
	if tokens[-1] not in ponct:
		tokens.append('.')
		period_added = True 
	return tokens, period_added



def accuracy_pos_tagging(ground_truth, prediction, period_added):
	"""
	Inputs :
	- the original parsing 
	- the result of the parser
	"""
	# extract the part of speech tags
	accuracy = 0
	length = 0
	pos_gt = []
	pos_pred = []
	tmp_gt = ground_truth.split(' ')
	tmp_pred = prediction.split(' ')
	for i, elt in enumerate(tmp_gt[:-1]):

		if elt[0] == '(' and tmp_gt[i + 1][-1] == ')':
			pos_gt.append(elt[1:])

	for i, elt in enumerate(tmp_pred[:-1]):
		if elt[0] == '(' and tmp_pred[i + 1][-1] == ')':
			pos_pred.append(elt[1:])

	if period_added:
		if len(pos_gt) == len(pos_pred) - 1:
			return float(np.sum([pos_gt[i] == pos_pred[:-1][i] for i in range(len(pos_gt))])) 
		else:
			return 0
	else:
		if len(pos_gt) == len(pos_pred):
			return float(np.sum([pos_gt[i] == pos_pred[i] for i in range(len(pos_gt))])) 
		else:
			return 0


	


if __name__ == '__main__':

	# fix seed 
	np.random.seed(456)
	############ train / development / evaluation split ###################
	with open("sequoia-corpus+fct.mrg_strict") as f:
		lines = f.readlines()

	# remove functional labels 
	permutation = np.random.permutation(range(len(lines)))

	train_indices = permutation[: int(len(permutation) * 0.8)]
	development_indices = permutation[int(len(permutation) * 0.8): int(len(permutation) * 0.9)]
	evaluation_indices = permutation[int(len(permutation) * 0.9): ]

	train_lines = preprocess([line for i, line in enumerate(lines) if i in train_indices])
	development_lines = preprocess([line for i, line in enumerate(lines) if i in development_indices])
	evaluation_lines = preprocess([line for i, line in enumerate(lines) if i in evaluation_indices])
	########################################################################


	# learn grammar 
	print("learning grammar ...")
	grammar = pcfg(train_lines)
	my_parser = probabilistic_parser(grammar, 'polyglot-fr.pkl')

	# evaluate and write results 
	f = open('evaluation_data.parser_output', "w")

	accuracies = []
	lengths = []
	for i, line in enumerate(evaluation_lines):
		
		tokens, period_added = tokenize(line)	
		print("parsing the sentence ... ;)")
		print("line", i,"-> " + line)
		parsing = '(' + my_parser.parse(tokens) +')'
		print(parsing)
		f.write(parsing)
		f.write('\n')

		accuracies.append(accuracy_pos_tagging(line, parsing, period_added))
		lengths.append(len(tokens) - float(period_added))
		accuracy = np.sum(np.array(accuracies)) / np.sum(np.array(lengths))
		print(f'accuracy after {i} steps : {accuracy}')

