import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import gc
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import math

import seaborn as sns
import matplotlib.pyplot as plt
from utils import *

# from bertkt_load_data import BertKTDataloader
torch.manual_seed(0)
np.random.seed(0)

class FFN(nn.Module):
	def __init__(self, ffn_dim, d_model):
		super(FFN, self).__init__()
		self.ffn_dim = ffn_dim
		self.d_model = d_model

		self.linear1 = nn.Linear(ffn_dim, ffn_dim)
		self.relu = nn.ReLU()
		self.linear2 = nn.Linear(ffn_dim, d_model)
		self.dropout = nn.Dropout(0.2)

	def forward(self, x):
		x = self.linear1(x)
		x = self.relu(x)
		x = self.linear2(x)
		x = self.dropout(x)
		return x

# def future_mask(seq_length):
#		 mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
#		 return torch.from_numpy(mask)

class DecoderLayer(nn.Module):
	def __init__(self, num_heads, d_model = 128):
		super(DecoderLayer, self).__init__()
		# self, input_dim, d_model, num_heads, dropout = 0
		self.att1 = MultiHeadAttention(input_dim = d_model, d_model=d_model, num_heads = num_heads, dropout = 0.2)
		self.att2 = MultiHeadAttention(input_dim = d_model, d_model=d_model, num_heads = num_heads, dropout = 0.2)
		self.ffn = FFN(d_model, d_model)

		self.layernorm1 = nn.LayerNorm(d_model)
		self.layernorm2 = nn.LayerNorm(d_model)
		self.layernorm3 = nn.LayerNorm(d_model)

	def forward(self, x, post_skill, look_ahead_mask):
		# device = x.device
		# x = x.permute(1, 0, 2)
		# post_skill = post_skill.permute(1, 0, 2)

		att_act1, att_w1 = self.att1(x, x, x, mask = look_ahead_mask)
		out1 = self.layernorm1(x + att_act1)
		# print('att score:')
		# print(att_w1)
		print('attended vec:')
		print(att_act1[0,:,:5])
		# print('layer normed vec:')
		# print(out1[0,:,:5])

		att_act2, att_w2 = self.att2(post_skill, out1, out1, mask = look_ahead_mask)
		out2 = self.layernorm2(out1 + att_act2)
		# out2 = out2.permute(1, 0, 2)
		# post_skill = post_skill.permute(1, 0, 2)

		ffn_act = self.ffn(out2)
		out3 = self.layernorm3(out2 + ffn_act)
		# print('att score:')
		# print(att_w2)
		print('attended vec:')
		print(att_act2[0,:,:5])
		# print('layer normed vec:')
		# print(out2[0,:,:5])
		# print('FFNed vec:')
		# print(ffn_act[0,:,:5])
		# print('layer normed vec:')
		# print(out3[0,:,:5])
		
		return out3, att_w1, att_w2

class DecoderLayerShuffle(nn.Module):
	def __init__(self, num_heads, d_model = 128):
		super(DecoderLayerShuffle, self).__init__()
		# self, input_dim, d_model, num_heads, dropout = 0
		self.att1 = MultiHeadAttentionShuffle(input_dim = d_model, d_model=d_model, num_heads = num_heads, dropout = 0.2)
		self.att2 = MultiHeadAttentionShuffle(input_dim = d_model, d_model=d_model, num_heads = num_heads, dropout = 0.2)
		self.ffn = FFN(d_model, d_model)

		self.layernorm1 = nn.LayerNorm(d_model)
		self.layernorm2 = nn.LayerNorm(d_model)
		self.layernorm3 = nn.LayerNorm(d_model)

	def forward(self, x, post_skill, look_ahead_mask):
		# device = x.device
		# x = x.permute(1, 0, 2)
		# post_skill = post_skill.permute(1, 0, 2)

		att_act1, att_w1 = self.att1(x, x, x, mask = look_ahead_mask)
		out1 = self.layernorm1(x + att_act1)

		att_act2, att_w2 = self.att2(post_skill, out1, out1, mask = look_ahead_mask)
		out2 = self.layernorm2(out1 + att_act2)
		# out2 = out2.permute(1, 0, 2)
		# post_skill = post_skill.permute(1, 0, 2)

		ffn_act = self.ffn(out2)
		out3 = self.layernorm3(out2 + ffn_act)
		
		return out3, att_w1, att_w2

# class PositionalEncoding(nn.Module):
# 	def __init__(self, d_model, dropout=0.1, max_len=5000):
# 		super(PositionalEncoding, self).__init__()
# 		self.dropout = nn.Dropout(p=dropout)

# 		pe = torch.zeros(max_len, d_model)
# 		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
# 		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
# 		pe[:, 0::2] = torch.sin(position * div_term)
# 		pe[:, 1::2] = torch.cos(position * div_term)
# 		# pe = pe.unsqueeze(0).transpose(0, 1)
# 		pe = pe.unsqueeze(0)
# 		self.register_buffer('pe', pe)

# 	def forward(self, x):
# 		x = x + self.pe[:, :x.shape[1], :]
# 		return self.dropout(x)

class BertKT(nn.Module):
	def __init__(self, num_layers, num_heads, n_prob, n_skill, d_model = 128, device_ls=None):
		super(BertKT, self).__init__()
		self.device_ls = device_ls
		self.n_skill = n_skill
		self.num_layers = num_layers
		self.emb_prior_rsps = nn.Embedding(2, d_model)
		self.emb_prior_skill = nn.Embedding(n_skill + 2, d_model)
		self.emb_post_skill = nn.Embedding(n_skill + 2, d_model)

		# self.pe = PositionalEncoding(d_model, dropout = 0.0, max_len = 200)

		self.dec_layers = []
		for i in range(num_layers):
			self.dec_layers.append(DecoderLayer(num_heads, d_model))
		self.dec_layers = nn.ModuleList(self.dec_layers)

	def future_mask(self, seq_length):
		mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
		return torch.from_numpy(mask)

	def forward(self, item, pt, ft, return_attn = False):
		
		if pt==1 and ft == 0:
			# prior_pid, prior_rsps, prior_skill, post_pid, post_rsps, post_skill, post_t, post_b #post_ski_index
			masked_skill, prior_skill, prior_rsps = item[0], item[2], item[3] # here we use the masked post pid (bs, 99)
			device = masked_skill.device
			prior_skill_vec = self.emb_prior_skill(prior_skill)
			prior_rsps_vec = self.emb_prior_rsps(prior_rsps)
			post_skill_vec = self.emb_post_skill(masked_skill)
			# look_ahead_mask = self.future_mask(masked_skill.size(1)).to(device)

			combo_vec = prior_skill_vec + prior_rsps_vec + post_skill_vec # + positional_encoding
			# combo_vec = self.pe(combo_vec)

			for i, dec_layer in enumerate(self.dec_layers):
				# combo_vec, _, _ = dec_layer(combo_vec, combo_vec, look_ahead_mask)
				combo_vec, _, _ = dec_layer(combo_vec, combo_vec, None)

			last_hs = combo_vec

			return last_hs

		if pt==0 and ft == 1:
			# old version: prior_pid, prior_rsps, prior_skill, post_pid, post_rsps, post_skill, post_t, post_b #post_ski_index
			# p, q, qa, t, b, att, hint, t_start, t_end, t_lag
			prior_skill, prior_rsps, post_skill = item[1][:,:-1], item[2][:,:-1], item[1][:,1:]
			device = prior_skill.device
			
			prior_skill_vec = self.emb_prior_skill(prior_skill) # (batch_size, seq_len, d_model)
			prior_rsps_vec = self.emb_prior_rsps(prior_rsps)
			post_skill_vec = self.emb_post_skill(post_skill)
			look_ahead_mask = self.future_mask(prior_skill.size(1)).to(device)

			combo_vec = prior_skill_vec + prior_rsps_vec + post_skill_vec # + positional_encoding
			print(combo_vec[0,:,:5])
			# combo_vec = self.pe(combo_vec)
			attn_ls = []
			for i, dec_layer in enumerate(self.dec_layers):
				combo_vec, attn_w1, attn_w2 = dec_layer(combo_vec, combo_vec, look_ahead_mask)
				# combo_vec, _, _ = dec_layer(combo_vec, combo_vec, None)
				attn_ls.append([attn_w1, attn_w2])
			last_hs = combo_vec
		if return_attn:
			return last_hs, attn_ls # n_decoder, 2, bs, n_head, seq_lem, seq_len
		else:
			return last_hs


class BertKTTestPermutation(nn.Module):
	def __init__(self, num_layers, num_heads, n_prob, n_skill, d_model = 128, device_ls=None):
		super(BertKTTestPermutation, self).__init__()
		self.device_ls = device_ls
		self.n_skill = n_skill
		self.num_layers = num_layers
		self.emb_prior_rsps = nn.Embedding(2, d_model)
		self.emb_prior_skill = nn.Embedding(n_skill + 2, d_model)
		self.emb_post_skill = nn.Embedding(n_skill + 2, d_model)

		# self.pe = PositionalEncoding(d_model, dropout = 0.0, max_len = 200)

		self.dec_layers = []
		for i in range(num_layers):
			self.dec_layers.append(DecoderLayer(num_heads, d_model))
		self.dec_layers = nn.ModuleList(self.dec_layers)

	def future_mask(self, seq_length):
		mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
		return torch.from_numpy(mask)

	def forward(self, item, pt, ft, return_attn = False):
		
		if pt==1 and ft == 0:
			# prior_pid, prior_rsps, prior_skill, post_pid, post_rsps, post_skill, post_t, post_b #post_ski_index
			masked_skill, prior_skill, prior_rsps = item[0], item[2], item[3] # here we use the masked post pid (bs, 99)
			device = masked_skill.device
			prior_skill_vec = self.emb_prior_skill(prior_skill)
			prior_rsps_vec = self.emb_prior_rsps(prior_rsps)
			post_skill_vec = self.emb_post_skill(masked_skill)
			# look_ahead_mask = self.future_mask(masked_skill.size(1)).to(device)

			combo_vec = prior_skill_vec + prior_rsps_vec + post_skill_vec # + positional_encoding
			# combo_vec = self.pe(combo_vec)

			for i, dec_layer in enumerate(self.dec_layers):
				# combo_vec, _, _ = dec_layer(combo_vec, combo_vec, look_ahead_mask)
				combo_vec, _, _ = dec_layer(combo_vec, combo_vec, None)

			last_hs = combo_vec

			return last_hs

		if pt==0 and ft == 1:
			# old version: prior_pid, prior_rsps, prior_skill, post_pid, post_rsps, post_skill, post_t, post_b #post_ski_index
			# p, q, qa, t, b, att, hint, t_start, t_end, t_lag
			prior_skill, prior_rsps, post_skill = item[0], item[1], item[2]
			device = prior_skill.device
			
			prior_skill_vec = self.emb_prior_skill(prior_skill) # (batch_size, seq_len, d_model)
			prior_rsps_vec = self.emb_prior_rsps(prior_rsps)
			post_skill_vec = self.emb_post_skill(post_skill)
			look_ahead_mask = self.future_mask(prior_skill.size(1)).to(device)

			combo_vec = prior_skill_vec + prior_rsps_vec + post_skill_vec # + positional_encoding
			# print(combo_vec[0,:,:5])
			# combo_vec = self.pe(combo_vec)
			attn_ls = []
			for i, dec_layer in enumerate(self.dec_layers):
				combo_vec, attn_w1, attn_w2 = dec_layer(combo_vec, combo_vec, look_ahead_mask)
				print('vec')
				print(combo_vec[0,:,:5])
				break
				# combo_vec, _, _ = dec_layer(combo_vec, combo_vec, None)
				attn_ls.append([attn_w1, attn_w2])
			last_hs = combo_vec
		if return_attn:
			return last_hs, attn_ls # n_decoder, 2, bs, n_head, seq_lem, seq_len
		else:
			return last_hs


class BertKTShuffle(nn.Module):
	def __init__(self, num_layers, num_heads, n_prob, n_skill, d_model = 128, device_ls=None):
		super(BertKTShuffle, self).__init__()
		self.device_ls = device_ls
		self.n_skill = n_skill
		self.num_layers = num_layers
		self.emb_prior_rsps = nn.Embedding(2, d_model)
		self.emb_prior_skill = nn.Embedding(n_skill + 2, d_model)
		self.emb_post_skill = nn.Embedding(n_skill + 2, d_model)

		# self.pe = PositionalEncoding(d_model, dropout = 0.0, max_len = 200)

		self.dec_layers = []
		for i in range(num_layers):
			self.dec_layers.append(DecoderLayerShuffle(num_heads, d_model))
		self.dec_layers = nn.ModuleList(self.dec_layers)

	def future_mask(self, seq_length):
		mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
		return torch.from_numpy(mask)

	def forward(self, item, pt, ft):
		
		if pt==1 and ft == 0:
			# prior_pid, prior_rsps, prior_skill, post_pid, post_rsps, post_skill, post_t, post_b #post_ski_index
			masked_skill, prior_skill, prior_rsps = item[0], item[2], item[3] # here we use the masked post pid (bs, 99)
			device = masked_skill.device
			prior_skill_vec = self.emb_prior_skill(prior_skill)
			prior_rsps_vec = self.emb_prior_rsps(prior_rsps)
			post_skill_vec = self.emb_post_skill(masked_skill)
			# look_ahead_mask = self.future_mask(masked_skill.size(1)).to(device)

			combo_vec = prior_skill_vec + prior_rsps_vec + post_skill_vec # + positional_encoding
			# combo_vec = self.pe(combo_vec)

			for i, dec_layer in enumerate(self.dec_layers):
				# combo_vec, _, _ = dec_layer(combo_vec, combo_vec, look_ahead_mask)
				combo_vec, _, _ = dec_layer(combo_vec, combo_vec, None)

			last_hs = combo_vec

			return last_hs

		if pt==0 and ft == 1:
			# old version: prior_pid, prior_rsps, prior_skill, post_pid, post_rsps, post_skill, post_t, post_b #post_ski_index
			# p, q, qa, t, b, att, hint, t_start, t_end, t_lag
			prior_skill, prior_rsps, post_skill = item[1][:,:-1], item[2][:,:-1], item[1][:,1:]
			device = prior_skill.device
			
			prior_skill_vec = self.emb_prior_skill(prior_skill) # (batch_size, seq_len, d_model)
			prior_rsps_vec = self.emb_prior_rsps(prior_rsps)
			post_skill_vec = self.emb_post_skill(post_skill)
			look_ahead_mask = self.future_mask(prior_skill.size(1)).to(device)

			combo_vec = prior_skill_vec + prior_rsps_vec + post_skill_vec # + positional_encoding

			# combo_vec = self.pe(combo_vec)
			for i, dec_layer in enumerate(self.dec_layers):
				combo_vec, _, _ = dec_layer(combo_vec, combo_vec, look_ahead_mask)
				# combo_vec, _, _ = dec_layer(combo_vec, combo_vec, None)

			last_hs = combo_vec

		return last_hs

class PT(nn.Module):
	def __init__(self, n_skill, d_model = 128, saved_bertkt = None, device_ls=None):
		super(PT, self).__init__()
		self.device_ls = device_ls
		self.saved_bertkt = saved_bertkt
		self.linear = nn.Linear(d_model, n_skill + 2)
	def forward(self, item, pt=1, ft = 0):
		last_hs = self.saved_bertkt(item, pt=pt, ft = ft)
		raw_pred = self.linear(last_hs)
		return raw_pred

class FT(nn.Module):
	def __init__(self, d_model = 128, saved_bertkt = None):
		super(FT, self).__init__()
		self.linear = nn.Linear(d_model, 1)
		self.saved_bertkt = saved_bertkt

	def forward(self, item, pt=0, ft=1, return_attn = False):
		if return_attn:
			last_hs, attn_ls = self.saved_bertkt(item, pt=pt, ft = ft, return_attn = return_attn)
			raw_pred = self.linear(last_hs)
			return raw_pred, attn_ls
		else:
			last_hs = self.saved_bertkt(item, pt=pt, ft = ft, return_attn = return_attn)
			raw_pred = self.linear(last_hs)
			return raw_pred


class FTTestPermutation(nn.Module):
	def __init__(self, d_model = 128, saved_bertkt = None):
		super(FTTestPermutation, self).__init__()
		self.linear = nn.Linear(d_model, 1)
		self.saved_bertkt = saved_bertkt

	def forward(self, prior_s, prior_a, post_s, pt=0, ft=1, return_attn = False):
		if return_attn:
			last_hs, attn_ls = self.saved_bertkt(item = (prior_s, prior_a, post_s), pt=pt, ft = ft, return_attn = return_attn)
			raw_pred = self.linear(last_hs)
			return raw_pred, attn_ls
		else:
			last_hs = self.saved_bertkt(item = (prior_s, prior_a, post_s), pt=pt, ft = ft, return_attn = return_attn)
			raw_pred = self.linear(last_hs)
			return raw_pred

'''================================Below is model builder testing code================================'''
# sample_ffn = FFN(32, 64)
# sample_ffn(torch.zeros((10,32))).shape

# sample_dec = DecoderLayer(num_heads = 2, d_model = 32)
# # sample_dec(torch.zeros((10,32,32), dtype = int), torch.zeros((10,32,32), dtype = int), None)
# sample_dec(torch.zeros((10,32,32)), torch.zeros((10,32,32)), None)[0].shape

# sample_bekt = BeKT(num_layers = 1, num_heads = 2, n_prob = 10, d_model = 32, seq_len = 10)
# sample_bekt(torch.zeros((10,32), dtype = int), torch.zeros((10,32), dtype = int)).shape

'''================================Below is model on real dataset testing code================================'''
# train_path = './edm_2022/data/assist_09/sequential/cv1.csv'
# val_path = './edm_2022/data/assist_09/sequential/cv2.csv'

# bertkt_dl = BertKTDataloader(train_path, val_path, val_path, group_path = None, shuffle = False, num_workers = 8)
# train_dataloader, val_dataloader, _, train_dataset, val_dataset, _ = bertkt_dl.get_data_loader()

# bertkt_ins = BertKT(num_layers = 2, num_heads = 4, n_prob = (35978+1), n_skill = (151+1))

# for item in train_dataloader:
#	 pred = bertkt_ins(item[0].long(), item[1].long(), item[2].long())
#	 print(pred.shape)
#	 break