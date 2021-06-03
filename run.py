import torch

import os
from os import path
import time
import random
import argparse

from data import *
from seq2seq import *

# preprocess
start_tok = '<SOS>'
end_tok = '<EOS>'
unk_tok = '<unk>'

# learning params
init_range = 0.08
epochs = 20
eval_step = 1000
batch_size = 64
criterion = nn.NLLLoss()

# seq2seq params
embedding_dim = 128
enc_hidden_dim = 128
dec_hidden_dim = 128

enc_layers = 1
dec_layers = 1

# output
model_path = "model.dat"

def evaluate(model, set):
	random.shuffle(set)
	preds = []
	for i in range(len(set)):
		#i=0
		lemma = set[i][0].tolist()
		word = set[i][1].tolist()
		feats = set[i][2].tolist()
		pred = model(set[i], type='evaluate')
		preds.append([word[1:], pred])
	return preds

def score(preds):
	correct = 0
	for pair in preds:
		# print('pair[0]', pair[0], 'pair[1]', pair[1])
		if pair[0] == pair[1]:
			correct += 1
	return correct / len(preds)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-lang', dest='lang', help='choose the language to run the task on', default='turkish', type=str)
	args = parser.parse_args()

	language = args.lang

	out_stats_path = 'results/' + language + '_seq2seq.tsv'

	# data paths
	train_path = 'data/conll2017/all/task1/' + language + '-train-high'
	dev_path = 'data/conll2017/all/task1/' + language + '-dev'
	test_path = 'data/conll2017/answers/task1/' + language + '-uncovered-test'

	data = Data(train_path, dev_path, test_path)
	char_vocab_len = data.create_char_vocab()
	feat_vocab_len = data.create_feat_vocab()
	train, dev, test = data.vectorize()
	# dimension of train is 10000 x 3 or 10000 x [lemma, word, feats]


	special_toks = {'sos': data.get_char_id(start_tok), 'eos': data.get_char_id(end_tok)}
	model = Seq2Seq(char_vocab_len, feat_vocab_len, embedding_dim, enc_hidden_dim, dec_hidden_dim, special_toks)

	if os.path.exists(model_path):
		saved_state = torch.load(model_path)
		model.load_state_dict(saved_state)
	else:
		model.init_weights(init_range)
		optimizer = optim.Adam(model.parameters())
		train_loss = 0
		step = 0

		train_holdout = train[:1000]

		with open(out_stats_path, 'w+') as f:
			f.write('epoch\ttrain_loss\ttrain_acc\tdev_acc\n')

		for epoch in range(epochs):
			random.shuffle(train)
			for i in range(len(train)):
				lemma = train[i][0]
				word = train[i][1]
				feats = train[i][2]
				pred = model(train[i], type='train')

				# print('word: {:30}   pred: {:30}'.format(data.vec2word(word), data.vec2word([x.max(0)[1].item() for x in pred])))

				total_loss = None
				# print('pred', pred.size(), 'word', word.size())
				loss = criterion(pred, word[1:])
				if total_loss is None:
					total_loss = loss
				else:
					total_loss += loss

				optimizer.zero_grad()
				total_loss.backward()# print('w_embeds_i =', w_embeds_i.size())
				# print('h0 =', h0.size())
				# print('self.c0 =', self.c0.size())
				optimizer.step()
				train_loss += total_loss

				step += 1
				if step % eval_step == 0:
					train_preds = evaluate(model, train_holdout)
					train_acc = score(train_preds)
					dev_preds = evaluate(model, dev)
					dev_acc = score(dev_preds)
					print('train examples')
					for j in range(5):
						print('word: {:30}   pred: {:30}'.format(data.vec2word(train_preds[j][0]), data.vec2word(train_preds[j][1])))
					print('dev examples')
					for j in range(5):
						print('word: {:30}   pred: {:30}'.format(data.vec2word(dev_preds[j][0]), data.vec2word(dev_preds[j][1])))
					print('epoch: {:.2f}/{:d}  completion: {:.2f}%  train loss: {:.4f}  train acc: {:f}  dev acc: {:f}'.format(
						float(epoch) + ((i+1)/len(train)), epochs, (step/(epochs*len(train)))*100, train_loss / eval_step, train_acc*100, dev_acc*100))

					train_loss_str = str(round((train_loss / eval_step).item(), 3))
					train_acc_str = str(round(train_acc*100, 2))
					dev_acc_str = str(round(dev_acc*100, 2))
					epoch_str = str(float(epoch) + ((i+1)/len(train)))
					with open(out_stats_path, 'a') as f:
						f.write(epoch_str + '\t' + train_loss_str + '\t' + train_acc_str + '\t' + dev_acc_str + '\n')

					print()
					train_loss = 0
