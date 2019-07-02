import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable



class Seq2Seq(nn.Module):
	def __init__(self, w_input_dim, f_input_dim, embed_dim, enc_hidden_dim, dec_hidden_dim, special_toks, batch_size=1, enc_layers=1, dec_layers=1, enc_num_directions=2):
		super(Seq2Seq, self).__init__()
		self.encoder = Encoder(w_input_dim, f_input_dim, embed_dim, enc_hidden_dim, batch_size, enc_layers, enc_num_directions=enc_num_directions)
		self.decoder = Decoder(w_input_dim, embed_dim, enc_hidden_dim, dec_hidden_dim, w_input_dim, batch_size, dec_layers, special_toks, enc_num_directions=enc_num_directions)
		#self.attention = nn.Linear()

	def init_weights(self, initrange):
		for param in self.parameters():
			param.data.uniform_(-initrange, initrange)

	def forward(self, input, type='train'):
		lemma = input[0]
		word = input[1]
		feats = input[2]
		#print('lemma',lemma)
		#print('feats',feats)

		hn = self.encoder(lemma ,feats)
		pred = self.decoder(lemma, word, hn, type=type)
		return pred

	def _fix_enc_hidden(hidden):
		# The encoder hidden is  (layers*directions) x batch x dim.
		# We need to convert it to layers x batch x (directions*dim).
		hidden = torch.cat([hidden[0:hidden.size(0):2],
							hidden[1:hidden.size(0):2]], 2)
		return hidden

class Encoder(nn.Module):
	def __init__(self, w_input_dim, f_input_dim, embed_dim, enc_hidden_dim, batch_size, num_layers, enc_num_directions=2):
		super(Encoder, self).__init__()

		self.w_input_dim = w_input_dim
		self.f_input_dim = f_input_dim
		self.embed_dim = embed_dim
		self.enc_hidden_dim = enc_hidden_dim
		self.batch_size = batch_size
		self.num_layers = num_layers
		self.enc_num_directions = enc_num_directions

		self.w_embedding = nn.Embedding(self.w_input_dim, self.embed_dim)
		self.f_embedding = nn.Embedding(self.f_input_dim, self.embed_dim)

		self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.enc_hidden_dim, num_layers=self.num_layers, bidirectional=(self.enc_num_directions == 2))
		self.h0 = Variable(torch.randn(self.enc_num_directions * self.num_layers, 1, self.enc_hidden_dim))
		self.c0 = Variable(torch.randn(self.enc_num_directions * self.num_layers, 1, self.enc_hidden_dim))

	def forward(self, lemma, feats):
		w_embeds = self.w_embedding(lemma)
		f_embeds = self.f_embedding(feats)
		embeds = torch.cat((w_embeds, f_embeds), 0).unsqueeze(1)
		out, (_, _) = self.lstm(embeds, (self.h0, self.c0))
		# print('out', out.size())
		# print('hn', hn.size())
		# print('cn', cn.size())
		return out[-1]

class Decoder(nn.Module):
	def __init__(self, w_input_dim, embed_dim, enc_hidden_dim, dec_hidden_dim, output_dim, batch_size, num_layers, special_toks, enc_num_directions=2):
		super(Decoder, self).__init__()

		self.w_input_dim = w_input_dim
		self.embed_dim = embed_dim
		self.enc_hidden_dim = enc_hidden_dim
		self.dec_hidden_dim = dec_hidden_dim
		self.output_dim = output_dim
		self.batch_size = batch_size
		self.num_layers = num_layers
		self.enc_num_directions = enc_num_directions

		self.start_tok = special_toks['sos']
		self.end_tok = special_toks['eos']

		self.w_embedding = nn.Embedding(self.w_input_dim, self.embed_dim)

		self.lstm = nn.LSTM(input_size=self.embed_dim + (enc_hidden_dim * enc_num_directions), hidden_size=self.enc_num_directions * self.dec_hidden_dim, num_layers=self.num_layers)
		self.c0 = Variable(torch.randn(self.num_layers, 1, self.enc_num_directions * self.dec_hidden_dim))

		self.linear = nn.Linear(self.enc_num_directions * self.dec_hidden_dim, self.output_dim)
		self.softmax = nn.LogSoftmax(dim=0)

	def forward(self, lemma, word, hn, type='train'):
		l_embeds = self.w_embedding(lemma)
		# h0 = torch.cat([hn[0], hn[1]], dim=1).unsqueeze(1)
		# state = (h0, self.c0)
		h0 = hn.squeeze()
		state = (hn.unsqueeze(1), self.c0)

		if type == 'train':
			pred = torch.zeros([len(word)-1, self.output_dim])

			w_embeds = self.w_embedding(word)

			for i in range(len(word)-1):
				w_embeds_i = w_embeds[i].view(1, 1, -1).squeeze()
				# print('w_embeds_i =', w_embeds_i.size())
				# print('h0 =', h0.size())
				# print('self.c0 =', self.c0.size())
				input = torch.cat([w_embeds_i, h0], dim=0).unsqueeze(0).unsqueeze(0)
				# print('input =', input.size())
				output, state = self.lstm(input, state)
				linear = self.linear(output.squeeze()).unsqueeze(1)
				# print('linear', linear.size())
				# print('self.output_dim', self.output_dim)
				softmax = self.softmax(linear).squeeze()
				# print('softmax.size()', softmax.size())
				for j in range(len(softmax)):
					pred[i][j] = softmax[j]
			# print('pred.size()', pred.size())

		elif type == 'evaluate':
			max_chars = len(lemma)*2 + 10
			#print(max_chars)

			start_tok_embed = self.w_embedding(torch.LongTensor([self.start_tok])).squeeze()
			end_tok_embed = self.w_embedding(torch.LongTensor([self.end_tok]))

			# print('start_tok_embed =', start_tok_embed.size())
			# print('h0 =', h0.size())
			# print('self.c0 =', self.c0.size())

			input = torch.cat([start_tok_embed, h0], dim=0).unsqueeze(0).unsqueeze(0)

			# print('input =', input.size())

			output, state = self.lstm(input, state)
			linear = self.linear(output.squeeze()).unsqueeze(1)
			softmax = self.softmax(linear).squeeze()
			pred = [softmax.max(0)[1].item()]
			i = 1

			#print('pred', pred[-1], 'start_tok',self.start_tok)
			while (not pred[-1] == self.end_tok) and i < max_chars:
				pred_embed = self.w_embedding(torch.LongTensor([pred[i-1]])).squeeze()
				input = torch.cat([pred_embed, h0], dim=0).unsqueeze(0).unsqueeze(0)
				output, state = self.lstm(input, state)
				linear = self.linear(output.squeeze()).unsqueeze(1)
				softmax = self.softmax(linear).squeeze()
				pred.append(softmax.max(0)[1].item())
				i += 1

		else:
			raise SystemExit('Error:', type, 'is invalid type (must be train or evaluate)')

		return pred
