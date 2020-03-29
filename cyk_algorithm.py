"""
implementation of cyk algorithm
"""

import numpy as np 
from tqdm import tqdm
from oov_module import oov_module


class probabilistic_parser():
	"""
	implementation of probabilistic parser object specific to a given grammar
	"""

	def __init__(self, grammar, embedding_file):
		self.grammar = grammar
		self.oov_module = oov_module(grammar, embedding_file, 4)


	def parse(self, tokens):
		processed, in_french_vocab = self.oov_module.handle(tokens)
		parsing = self.probabilistic_CYK(processed, tokens, in_french_vocab)
		return self.remove_chomsky_symbols(parsing)


	def remove_chomsky_symbols(self, parsed_sentence):
		tmp = parsed_sentence.split(' ')
		processed = tmp.copy()
		removed = 0
		for i, elt in enumerate(tmp):
			if '|' in elt:
				opening = 1
				closing = 0
				for j, rest in enumerate(tmp[i+1:]):
					if '(' in rest:
						opening += 1
					else:
						for s in rest:
							if s == ')':
								closing += 1
				
					if closing == opening:
						processed[j + i + 1 - removed] = processed[j + i + 1 - removed][:-1]
						del processed[i - removed]
						break
				removed += 1
		return ' '.join(processed)


	def build_tree(self, back, originals, table):
		"""
		build the parsing tree of the sentence from the returned argument of probabilistic cyk
		based on the recursive unfolding of back
		"""

		def unfold(node, i, j, const, back, idx, originals):
			"""
			recursive function 
			"""
			# stop condition
			if node == -1:
				word = originals[i]
				return const + ' ' + word 
			return const + ' (' + unfold(back[i, node[0], idx[node[1][0]]], i, node[0], node[1][0], back, idx, originals) + ') (' + unfold(back[node[0], j, idx[node[1][1]]], node[0], j, node[1][1], back, idx, originals) + ')'

		n = len(originals)
		
		return '(' + unfold(back[0, n, self.grammar.idx[self.grammar.start_symbol]], 0, n, self.grammar.start_symbol, back, self.grammar.idx, originals) + ')'


	def probabilistic_CYK(self, tokens, originals, in_french_vocab):
		"""
		Inspired from the pseudo code in the Stanford NLP course chapter 14 
		tokens : tokenized sentence 
		returns most probable parse

		assumed the grammar is in chomsky normal form 
		and the pb of oov token is handled (i.e each token in tokens is in our vocabulary)
		"""

		n = len(tokens)

		# table : table of size (n) * (n) * (nbr_of_terminal_symbols + nbr_of_non_terminal_symbols) (assumed to be known as global variables or elements of the class parser)
		# table[i, j, k] contains the probabilty of the sequence from i to j to be a constituent of type k
		# back is an array of backpointers used to recover the best parse. 
		# i.e : back[i, j, k] contains (l, m, n) s.t k -> mn

		table = np.zeros((n + 1, n + 1, self.grammar.nbr_ts + self.grammar.nbr_nts))
		back = - np.ones((n + 1, n + 1, self.grammar.nbr_ts + self.grammar.nbr_nts), dtype='O')

		# complete the table in a bottom up fashion : each column from bottom to top and from left to right 
		for j in range(1, n + 1):
			print(f"processing token {j}")
			# super diagonal 
			for ts in self.grammar.terminal_symbols:
				if (ts, tokens[j - 1]) in self.grammar.lexicons:
					table[j - 1, j, self.grammar.idx[ts]] = self.grammar.lexicons[(ts, tokens[j - 1])]

				# handle proper nouns 
				if originals[j - 1][0].isupper() and table[j - 1, j, self.grammar.idx["NP"]] == 0 and table[j - 1, j, self.grammar.idx["NPP"]] == 0 and not(in_french_vocab[j - 1]):
					# hyperparameter 
					table[j - 1, j, self.grammar.idx["NP"]] = 0.0001
					table[j - 1, j, self.grammar.idx["NPP"]] = 0.0001

			# rest of the column 

			for i in range(j - 2, -1, -1):
				
				for k in range(i + 1, j):
					
					for nt in self.grammar.non_terminal_symbols:
						right_side = self.grammar.right_side[nt]
						for (b, c) in right_side:
							if table[i, k, self.grammar.idx[b]] > 0 and table[k, j, self.grammar.idx[c]] > 0:
								if table[i, j, self.grammar.idx[nt]] < self.grammar.rules[(nt, (b, c))] * table[i, k, self.grammar.idx[b]] * table[k, j, self.grammar.idx[c]]:
									table[i, j, self.grammar.idx[nt]] = self.grammar.rules[(nt, (b, c))] * table[i, k, self.grammar.idx[b]] * table[k, j, self.grammar.idx[c]]
									back[i, j, self.grammar.idx[nt]] = (k, (b, c))
										
		return self.build_tree(back, originals, table)


	# def beam_search(self, tokens, originals, beam_size):
	# 	"""
	# 	"""

	# 	def find_best(probs, beam_size):
	# 		"""
	# 		"""
	# 		best = sorted(probs.items(), key=lambda kv: kv[1])[:beam_size]
	# 		table, back = [(elt[0][0], elt[1]) for elt in best], [(elt[0][1], elt[0][2], elt[0][3]) for elt in best]
	# 		return (table, back)

	# 	n = len(tokens)

	# 	table = np.zeros((n + 1 , n + 1, beam_size), dtype = 'O')
	# 	back = -np.ones((n + 1 , n + 1, beam_size), dtype = 'O')

	# 	for j in range(1, n + 1):
	# 		print(f"processing token {j}")

	# 		pos_list = self.grammar.token_to_pos[tokens[j - 1]][:min(len(self.grammar.token_to_pos[tokens[j - 1]]),beam_size)]
	# 		table[j - 1, j, :min(beam_size, len(pos_list))] = pos_list

	# 		for i in range(j - 2, -1, -1):
	# 			total_probs = {}
	# 			for k in range(i + 1, j):
	# 				for p in range(beam_size):
	# 					for q in range(beam_size):

	# 						if table[i, k, p] != 0 and table[k, j, q] != 0:
	# 							probs = {}
	# 							b, p_b, c, p_c = table[i, k, p][0], table[i, k, p][1], table[k, j, q][0], table[k, j, q][1]
	# 							if (b, c) in self.grammar.prec:
	# 								prec = self.grammar.prec[(b, c)][: min(len(self.grammar.prec[b,c]), beam_size)]
	# 								print(b, c, prec)
	# 								for (nt, prob) in prec:
	# 									probs[(nt, p, q, k)] = prob * p_b * p_c

	# 			total_probs.update(probs)

	# 			best = find_best(total_probs, beam_size)
	# 			table[i, j, :min(beam_size, len(best[0]))] = best[0]
	# 			back[i, j, :min(beam_size, len(best[0]))] = best[1]

	# 	return table, back




            