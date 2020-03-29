import argparse
import numpy as np
import re 

from preprocessing import preprocess 
from extract_pcfg import pcfg
from cyk_algorithm import probabilistic_parser

parser = argparse.ArgumentParser()
parser.add_argument("--lines", type=str, nargs='+')
args = parser.parse_args()

ponct = ['.', '!', '?', '...', ':', ';', ',']

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

	output_path = './output.txt'
	f = open(output_path, 'w')
	print("parsing inputs : ...")
	lines = []
	tmp = []
	for i, token in enumerate(args.lines):
		if token != '\n':
			tmp.append(token)
		else:
			lines.append(' '.join(tmp))
			tmp = []
		if i == len(args.lines) - 1 and token != '\n':
			lines.append(' '.join(tmp))
	
	for line in lines:
		tokens = line.split(' ')
		if tokens[-1] not in ponct:
			tokens.append('.')
		else:
			print("parsing sentence ... ;)")
			print("line :", line)
			parsing = '(' + my_parser.parse(tokens) +')'
			print(parsing)
			f.write(parsing)
			f.write('\n')

