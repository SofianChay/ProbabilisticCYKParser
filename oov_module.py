"""
oov module to handle spelling errors and genuine unknown words
"""

import pickle 
import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm
from language_model import language_model 
from sklearn.neighbors import NearestNeighbors

class oov_module():

	def __init__(self, grammar, embedding_file, k):

		self.grammar_words = list(tk for _, tk in grammar.lexicons)
		self.french_vocabulary, self.french_embeddings = pickle.load(open(embedding_file, 'rb'), encoding='latin1')
		self.language_model = language_model(grammar.sentences)

		self.vocabulary, self.embedding = [w for w in self.french_vocabulary if w in self.grammar_words], [vec for i, vec in enumerate(self.french_embeddings) if self.french_vocabulary[i] in self.grammar_words]

		self.neigh = NearestNeighbors(k)
		self.neigh.fit(self.embedding)

		self.idx = {w: i for i, w in enumerate(self.french_vocabulary)}


	def most_similar_embeddings(self, token):
		"""
		return the k most similar words in the lexicon according to the embedding 
		"""
		most_similar = [self.vocabulary[i] for i in self.neigh.kneighbors(self.french_embeddings[self.idx[token]].reshape(1, -1))[1][0]]
		return most_similar


	def levenstein(self, string1, string2):
		"""
		return the levenstein distance between string1 and string2
		dynamic programming approach 
		"""
		n = len(string1)
		m = len(string2)
		if min(n, m) == 0:
			return max(n, m)

		table = np.zeros((n, m), dtype=int)
		for i in range(n):
			for j in range(m):
				if min(i, j) == 0:
					table[i, j] = max(i, j)
				else:
					table[i, j] = min([table[i, j - 1] + 1, table[i - 1, j] + 1, table[i - 1, j - 1] + int(string1[i] != string2[j])])
		return table[n - 1, m - 1]


	def most_similar_levenstein(self, token):
		"""
		return the words of the french vocab within a minimal levenstein distance from the string 
		"""
		distances = [self.levenstein(word, token) for word in self.french_vocabulary]
		mini = min(distances)
		closest =[w for i, w in enumerate(self.french_vocabulary) if distances[i] == mini]
		return closest


	def most_probable_bigram(self, prev_token, candidates):
		"""
		given a list of substitute candidates it returns the most probable one given the previous token 
		warning : the prev_token needs to be in the lexicon as well as the candidates
		"""
	
		best = candidates[0]
		prob = self.language_model.matrix[self.language_model.index_vocabulary[prev_token], self.language_model.index_vocabulary[candidates[0]]]
		for candidate in candidates[1:]:
			tmp = self.language_model.matrix[self.language_model.index_vocabulary[prev_token], self.language_model.index_vocabulary[candidate]]
			if prob < tmp:
				prob = tmp
				best = candidate
		return best


	def most_probable_unigram(self, candidates):
		best = candidates[0]
		prob = self.language_model.unigram[self.language_model.index_vocabulary[candidates[0]]]
		for candidate in candidates[1:]:
			tmp = self.language_model.unigram[self.language_model.index_vocabulary[candidate]]
			if prob < tmp:
				prob = tmp
				best = candidate
		return best

	
	def handle(self, tokens):
		"""
		given a sentence, returns a substitue sentence such that each token is in the lexicon
		"""
		processed = []
		in_french_vocab = []
		for i, token in enumerate(tokens):
			if token.lower() in self.vocabulary:
				processed.append(token.lower())
				in_french_vocab.append(True)

			elif token.lower() in self.french_vocabulary:
				in_french_vocab.append(True)
				candidates = self.most_similar_embeddings(token.lower())
				if i == 0:
					processed.append(self.most_probable_unigram(candidates))
				else:
					processed.append(self.most_probable_bigram(processed[i - 1], candidates))

			else:
				in_french_vocab.append(False)
				closest = self.most_similar_levenstein(token.lower())
				candidates = []
				for w in closest:
					candidates += self.most_similar_embeddings(w)
				if i == 0:
					processed.append(self.most_probable_unigram(candidates))
				else:
					processed.append(self.most_probable_bigram(processed[i - 1], candidates))

		return processed, in_french_vocab

	