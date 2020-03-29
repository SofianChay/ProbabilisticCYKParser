"""
Extract pcfg from training set
"""

import numpy as np
import nltk 
from nltk import Tree
import re 
from tqdm import tqdm 

class pcfg():

	def __init__(self, lines):
		"""
		class that takes parsed sentences assumed to be correctly preprocessed as input 
		"""

		def extract_terminal_non_terminal_symbols(lines):
			terminals = set()
			non_terminals = set()

			for line in lines:
				tmp = line.split(' ')
				
				for i, elt in enumerate(tmp[: -1]):
				
					if tmp[i + 1][0] == '(' and elt[-1] != ')':
						non_terminals.add(elt[1:])
					elif elt[0] == '(':
						terminals.add(elt[1:])
					
			return terminals, non_terminals


		def extract_lexicon(lines):
			lexicon = dict()

			for line in lines:
				tmp = line.split(' ')
				for i ,w in enumerate(tmp):
					if w[-1] == ')':
						j = -2
						while -j <= len(w) and w[j] == ')':
							j -= 1
						token = w[:j + 1].lower()
						pos_tag = tmp[i - 1][1:]
						if (pos_tag, token) not in lexicon:
							lexicon[(pos_tag, token)] = 1
						else:
							lexicon[(pos_tag, token)] += 1

			grouped_by_tk = dict() 
			for pos, tk in lexicon:
				if tk not in grouped_by_tk:
					grouped_by_tk[tk] = [lexicon[(pos, tk)]]
				else:
					grouped_by_tk[tk].append(lexicon[(pos, tk)])

			for pos, tk in lexicon:
				lexicon[(pos, tk)] = lexicon[(pos, tk)] / np.sum(grouped_by_tk[tk])

			return lexicon


		def extract_rules(lines):
			rules = []
			for line in lines:
				count_opened = 0
				count_closed = 0

				tmp = line.split(' ')
				sent_rules = [] # list of lists of rules 
				closed_rule = [] # list of bools

				for i, elt in enumerate(tmp[: -1]):
					if tmp[i + 1][0] == '(':
						count_opened += 1
						
						pointer_level = count_opened - count_closed - 1
						if len(sent_rules) >= pointer_level + 1:
							if closed_rule[pointer_level] and elt[-1] != ')':
								sent_rules[pointer_level].append((elt[1:], [tmp[i + 1][1:]]))
							else:

								sent_rules[pointer_level][-1][1].append(tmp[i + 1][1:])

						else:
							if elt[-1] != ')':
								sent_rules.append([(elt[1:], [tmp[i + 1][1:]])])
								closed_rule.append(False)

					elif tmp[i + 1][-1] == ')':
						j = -1
						while -j + 1 <= len(tmp[i + 1]) and tmp[i + 1][j - 1] == ')':
							j -= 1
						count_closed -= j

					if count_closed == count_opened:
						closed_rule[pointer_level] = True
	 					
				rules += [rule for level in sent_rules for rule in level]

			dict_rules = {}
			for i in range(len(rules)):
				rules[i] = (rules[i][0], tuple(rules[i][1]))
			for rule in rules:
				if rule in dict_rules:
					dict_rules[rule] += 1
				else:
					dict_rules[rule] = 1

			return dict_rules


		def remove_unit_productions(rules, terminal_symbols):
			"""
			removed unit productions rules 
			to do before binarization
			"""

			result = {}
			unitary = []

			for rule in rules:
				if len(rule[1]) == 1 and rule[1] not in terminal_symbols:
					unitary.append((rule, rules[rule]))
				else:
					result[rule] = rules[rule]

			while unitary:
				(rule, value) = unitary.pop(0)
				right_side = [r[1] for r in rules if r[0] == rule[1]]
				for item in right_side:
					new_rule = (rule[0], item)
					if len(item) != 1 or item in terminal_symbols:
						result[new_rule] = value
					else:
						unitary.append((new_rule, value))


			return result


		def binarize(rules):
			"""
			rules : dict of rules 
			return : binarized : dict of probabilistic rules in binarized form 
			"""
			binarized = {}

			def transform(rule, value):
				transformed_rules = {}
				cnt = len(rule[1])
				unprocessed = list(rule[1])
				left_side = rule[0]
				while cnt >= 2:
					new_constituent = '|'.join(unprocessed[: -1])
					transformed_rules[(left_side, (new_constituent, unprocessed[-1]))] = value
					left_side = new_constituent
					unprocessed.pop()
					cnt -= 1
				return transformed_rules

			for rule in rules:
				if len(rule[1]) == 2:
					binarized[rule] = rules[rule]

				else:
					binarized.update(transform(rule, rules[rule]))
					
			return binarized


		def probabilistic(rules):
			pcfg = {}
			grouped_by = dict() 
			for left, right in rules:
				if left not in grouped_by:
					grouped_by[left] = [rules[(left, right)]]
				else:
					grouped_by[left].append(rules[(left, right)])

			for left, right in rules:
				pcfg[(left, right)] = rules[(left, right)] / np.sum(grouped_by[left])

			return pcfg


		def extract_non_terminals_from_rules(rules):
			non_terminals = set()
			for rule in rules:
				non_terminals.add(rule[0])
			return non_terminals

		self.start_symbol = 'SENT'
		self.terminal_symbols, _ = extract_terminal_non_terminal_symbols(lines)
		self.lexicons = extract_lexicon(lines)
		self.rules = probabilistic(binarize(remove_unit_productions(extract_rules(lines), self.terminal_symbols)))
		self.non_terminal_symbols = extract_non_terminals_from_rules(self.rules)
		self.idx_non_terminal = {nt: i for i, nt in enumerate(self.non_terminal_symbols)}
		self.idx_terminal = {t: len(self.non_terminal_symbols) + i for i, t in enumerate(self.terminal_symbols)}
		self.idx = {**self.idx_terminal, **self.idx_non_terminal}
		self.index_to_sym = {i: sym for sym, i in self.idx.items()}
		self.nbr_ts = len(self.terminal_symbols)
		self.nbr_nts = len(self.non_terminal_symbols)


		def inverse_lexicon(lexicons):
			"""
			"""
			token_to_pos = {}

			for pos, tk in lexicons:
				if tk not in token_to_pos:
					token_to_pos[tk] = [(pos,lexicons[(pos, tk)])]
				else:
					token_to_pos[tk].append((pos,lexicons[(pos, tk)]))

			# sort 

			for tk in token_to_pos:
				token_to_pos[tk] = sorted(token_to_pos[tk], key=lambda t: t[1])

			return token_to_pos


		self.token_to_pos = inverse_lexicon(self.lexicons)


		def build_prec(rules):
			"""
			"""
			prec = {}

			for rule in rules:
				if rule[1] not in prec:
					prec[rule[1]] = [(rule[0], rules[rule])]
				else:
					prec[rule[1]].append((rule[0], rules[rule]))

			# sort 
			
			for left_side in prec:
				prec[left_side] = sorted(prec[left_side], key=lambda t: t[1])
			return prec 


		self.prec = build_prec(self.rules)


		def extract_sentences(lines):
			processed = []
			for line in lines:
				sentence = []
				tmp = line.split(' ')
				for elt in tmp:
					if elt[-1] == ')':
						word = ''
						j = 0
						while elt[j] != ')':
							word += elt[j]
							j += 1
						sentence.append(word[:j].lower())
				processed.append(sentence)
			return processed
	

		self.sentences = extract_sentences(lines)


		def rightSide(rules):
			right_side = {}
			for rule in rules:
				if rule[0] not in right_side:
					right_side[rule[0]] = [rule[1]]
				else:
					right_side[rule[0]].append(rule[1])

			return right_side

		self.right_side = rightSide(self.rules)


	def rightSide(self, nt):
		right_side = []
		for rule in self.rules:
			if rule[0] == nt:
				right_side.append(rule[1])
		return right_side
