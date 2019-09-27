import torch
from run import *

class Dataset:
	lemmas = []
	words = []
	feats = []

	def __init__(self, lemmas, words, feats):
		self.lemmas = lemmas
		self.words = words
		self.feats = feats

	def vectorize(self, char_vocab, feat_vocab):
		lemmas = []
		words = []
		feats = []

		for lemma in self.lemmas:
			word_vec = [char_vocab[start_tok]]
			for char in lemma:
				if char in char_vocab.keys():
					word_vec.append(char_vocab[char])
				else:
					word_vec.append(char_vocab[unk_tok])
			word_vec.append(char_vocab[end_tok])
			lemmas.append(torch.tensor(word_vec))

		for word in self.words:
			word_vec = [char_vocab[start_tok]]
			for char in word:
				if char in char_vocab.keys():
					word_vec.append(char_vocab[char])
				else:
					word_vec.append(char_vocab[unk_tok])
			word_vec.append(char_vocab[end_tok])
			words.append(torch.tensor(word_vec))

		for featslist in self.feats:
			feats_vec = []
			for feat in featslist:
				if feat in feat_vocab.keys():
					feats_vec.append(feat_vocab[feat])
				else:
					feats_vec.append(feat_vocab[unk_tok])
			feats.append(torch.tensor(feats_vec))

		retlist = []

		for i in range(len(lemmas)):
			retlist.append([lemmas[i], words[i], feats[i]])

		return retlist

class Data:
	char_vocab = {}
	feat_vocab = {}

	def __init__(self, train_path, dev_path, test_path):
		self.train = self.loadData(train_path)
		self.dev = self.loadData(dev_path)
		self.test = self.loadData(test_path)

	def batchify(self, size):
		train_batches = []
		dev_batches = []
		test_batches = []

		train_sort = self.train.sort()

	def loadData(self, path):
		lemmas = []
		words = []
		feats = []

		with open(path, 'r') as file:
			data = [line.strip().split('\t') for line in file]
		for line in data:
			lemmas.append(line[0])
			words.append(line[1])
			feats.append(line[2].split(';'))

		return Dataset(lemmas, words, feats)

	def create_char_vocab(self):
		char_set = set()

		for word in self.train.lemmas:
			char_set |= set(word)
		for word in self.train.words:
			char_set |= set(word)

		#print('char_set:', char_set)

		for i, char in enumerate(char_set):
			self.char_vocab[char] = i

		self.char_vocab[start_tok] = i + 1
		self.char_vocab[end_tok] = i + 2
		self.char_vocab[unk_tok] = i + 3

		return len(self.char_vocab.keys())

	def create_feat_vocab(self):
		feat_set = set()

		for featslist in self.train.feats:
			for feat in featslist:
				feat_set.add(feat)

		for i, feat in enumerate(feat_set):
			self.feat_vocab[feat] = i

		self.feat_vocab[unk_tok] = i + 1

		return len(self.feat_vocab.keys())

	def vectorize(self):
		if not self.char_vocab or not self.feat_vocab:
			raise SystemExit('Error: need to initialize char_vocab and feat_vocab first!')

		train_vecs = self.train.vectorize(self.char_vocab, self.feat_vocab)
		dev_vecs = self.dev.vectorize(self.char_vocab, self.feat_vocab)
		test_vecs = self.test.vectorize(self.char_vocab, self.feat_vocab)

		return train_vecs, dev_vecs, test_vecs

	def vec2word(self, vec):
		word = ''
		for x in vec:
			for k, v in self.char_vocab.items():
				if x == v:
					word += k
					break
		return word

	def vec2featlist(self, vec):
		featlist = []
		for x in vec:
			for k, v in self.feat_vocab.items():
				if x == v:
					featlist.append(k)
					break
		return featlist

	def get_char_id(self, char):
		if char in self.char_vocab.keys():
			return  self.char_vocab[char]
		return None
