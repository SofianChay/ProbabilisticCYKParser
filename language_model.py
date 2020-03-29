"""
language model object based on bigrams 
"""

import numpy as np 


class language_model():
	

	def __init__(self, lines):

		def build_vocab(lines):
			vocab = set()
			for line in lines:
				for w in line:
					vocab.add(w.lower())
			return vocab

		self.vocabulary = build_vocab(lines)

		self.index_vocabulary = {word: i for i, word in enumerate(self.vocabulary)}
		def build_language_model(lines, vocabulary, index_vocabulary):
			counts = {}
			pairs = {}

			transition_matrix = np.zeros((len(vocabulary), len(vocabulary))) 
			unigram = np.zeros(len(vocabulary))
			for line in lines:
				for i, word in enumerate(line):
					if i != 0:
						j = index_vocabulary[word]
						k = index_vocabulary[line[i- 1]]
						if j not in counts:
							counts[j] = 1
						else:
							counts[j] += 1
						if (k, j) not in pairs:
							pairs[(k, j)] = 1
						else:
							pairs[(k, j)] += 1

			for (i, j) in pairs:
			  transition_matrix[i, j] = pairs[(i, j)] / counts[j]

			for i in counts:
				unigram[i] = counts[i] / sum(counts.values())

			return transition_matrix, unigram

		self.matrix, self.unigram = build_language_model(lines, self.vocabulary, self.index_vocabulary)

