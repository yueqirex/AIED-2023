import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat, randperm

import numpy as np
import pandas as pd
import math
import os
torch.manual_seed(0)
np.random.seed(0)

def get_path(model_name, data_base, ckpt_base):
	cv_path_ls = []
	ckpt_path_ls = []
	for i in range(5):
		train_path = os.path.join(data_base, 'cv{}_train.csv'.format(i))
		val_path = os.path.join(data_base, 'cv{}_val.csv'.format(i))
		test_path = os.path.join(data_base, 'cv{}_test.csv'.format(i))
		cv_path_ls.append([train_path, val_path, test_path])

	if model_name in ['dkt', 'DKT']:
		for i in range(5):
			ckpt_path_ls.append(os.path.join(ckpt_base, 'cv{}.pt'.format(i)))
	else:
		raise Exception('model name is wrong, current model_name:{}'.format(model_name))
	
	return cv_path_ls, ckpt_path_ls



def get_device_names(device_group, parallel = False):
	if parallel == False:
		return device_group # this is equal to device id itself

	else:
		if device_group == 'cpu':
			device_ids = None
			device_names = ['cpu', 'cpu']
		elif device_group == 'cuda:0':
			device_ids = None
			device_names = ['cuda:0', 'cuda:0']
		elif device_group == 'cuda:1':
			device_ids = None
			device_names = ['cuda:1', 'cuda:1']
		elif device_group == 'cuda:2':
			device_ids = None
			device_names = ['cuda:2', 'cuda:2']
		elif device_group == 'cuda:3':
			device_ids = None
			device_names = ['cuda:3', 'cuda:3']
		elif device_group == 0:
			device_ids=[0,1]
			device_names = ['cuda:0','cuda:1']
		elif device_group == 1:
			device_ids=[2,3]
			device_names = ['cuda:2','cuda:3']
		elif device_group == 2:
			device_ids=[4,5]
			device_names = ['cuda:4','cuda:5']
		elif device_group == 3:
			device_ids=[6,7]
			device_names = ['cuda:6','cuda:7']

		elif device_group == 4:
			device_ids=[0,1,2,3]
			device_names = ['cuda:0','cuda:1','cuda:2','cuda:3']
		elif device_group == 5:
			device_ids=[4,5,6,7]
			device_names = ['cuda:4','cuda:5','cuda:6','cuda:7']
		elif device_group == 6:
			device_ids=[0,1,2,3,4,5]
			device_names = ['cuda:0','cuda:1','cuda:2','cuda:3',\
							'cuda:4','cuda:5']
		# print('---NOTE--- Training devices: ', device_names)
		return device_ids, device_names




def get_path(model_name, data_base, ckpt_base):
	cv_path_ls = []
	ckpt_path_ls = []
	for i in range(5):
		train_path = os.path.join(data_base, 'cv{}_train.csv'.format(i))
		val_path = os.path.join(data_base, 'cv{}_val.csv'.format(i))
		test_path = os.path.join(data_base, 'cv{}_test.csv'.format(i))
		cv_path_ls.append([train_path, val_path, test_path])

	if model_name in ['dkt', 'DKT']:
		for i in range(5):
			ckpt_path_ls.append(os.path.join(ckpt_base, 'pt_cv{}.pt'.format(i)))
	elif model_name in ['ode_dkt', 'ode_DKT']:
		for i in range(5):
			ckpt_path_ls.append(os.path.join(ckpt_base, 'ft_cv{}.pt'.format(i)))
	else:
		raise Exception('model name is wrong, current model_name:{}'.format(model_name))
	
	return cv_path_ls, ckpt_path_ls





def preview_dataloader_new(dataloader_ls):
	train_dataloader, val_dataloader, test_dataloader = dataloader_ls[0], dataloader_ls[1], dataloader_ls[2]
	for i, item in enumerate(train_dataloader):
		print('train skill_rsps_hots shape: ', item['skill_rsps_hots'].shape)
		print('train skill_hots shape: ', item['skill_hots'].shape)
		print('train rsps_array shape: ', item['rsps_array'].shape)
		print('train skill_array shape: ', item['skill_array'].shape)
		print('train timesteps shape: ', item['timesteps'].shape)
		print('==============================================================')
		break
	for i, item in enumerate(val_dataloader):
		print('val batch shape: ', item['skill_rsps_hots'].shape)
		break
	for i, item in enumerate(test_dataloader):
		print('test batch shape: ', item['skill_rsps_hots'].shape)
		break
	return



def preview_dataloader(dataloader_ls):
	train_dataloader, val_dataloader, test_dataloader = dataloader_ls[0], dataloader_ls[1], dataloader_ls[2]
	for i, item in enumerate(train_dataloader):
		print('train batch shape: ', item[0].shape)
		break
	for i, item in enumerate(val_dataloader):
		print('val batch shape: ', item[0].shape)
		break
	for i, item in enumerate(test_dataloader):
		print('test batch shape: ', item[0].shape)
		break



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat, randperm

import numpy as np
import pandas as pd
import math
import os
torch.manual_seed(0)
np.random.seed(0)

def get_path(model_name, data_base, ckpt_base):
	cv_path_ls = []
	ckpt_path_ls = []
	for i in range(5):
		train_path = os.path.join(data_base, 'cv{}_train.csv'.format(i))
		val_path = os.path.join(data_base, 'cv{}_val.csv'.format(i))
		test_path = os.path.join(data_base, 'cv{}_test.csv'.format(i))
		cv_path_ls.append([train_path, val_path, test_path])

	if model_name in ['dkt', 'DKT']:
		for i in range(5):
			ckpt_path_ls.append(os.path.join(ckpt_base, 'cv{}.pt'.format(i)))
	else:
		raise Exception('model name is wrong, current model_name:{}'.format(model_name))
	
	return cv_path_ls, ckpt_path_ls



def get_device_names(device_group, parallel = False):
	if parallel == False:
		return device_group # this is equal to device id itself

	else:
		if device_group == 'cpu':
			device_ids = None
			device_names = ['cpu', 'cpu']
		elif device_group == 'cuda:0':
			device_ids = None
			device_names = ['cuda:0', 'cuda:0']
		elif device_group == 'cuda:1':
			device_ids = None
			device_names = ['cuda:1', 'cuda:1']
		elif device_group == 'cuda:2':
			device_ids = None
			device_names = ['cuda:2', 'cuda:2']
		elif device_group == 'cuda:3':
			device_ids = None
			device_names = ['cuda:3', 'cuda:3']
		elif device_group == 0:
			device_ids=[0,1]
			device_names = ['cuda:0','cuda:1']
		elif device_group == 1:
			device_ids=[2,3]
			device_names = ['cuda:2','cuda:3']
		elif device_group == 2:
			device_ids=[4,5]
			device_names = ['cuda:4','cuda:5']
		elif device_group == 3:
			device_ids=[6,7]
			device_names = ['cuda:6','cuda:7']

		elif device_group == 4:
			device_ids=[0,1,2,3]
			device_names = ['cuda:0','cuda:1','cuda:2','cuda:3']
		elif device_group == 5:
			device_ids=[4,5,6,7]
			device_names = ['cuda:4','cuda:5','cuda:6','cuda:7']
		elif device_group == 6:
			device_ids=[0,1,2,3,4,5]
			device_names = ['cuda:0','cuda:1','cuda:2','cuda:3',\
							'cuda:4','cuda:5']
		# print('---NOTE--- Training devices: ', device_names)
		return device_ids, device_names




def get_path(model_name, data_base, ckpt_base):
	cv_path_ls = []
	ckpt_path_ls = []
	for i in range(5):
		train_path = os.path.join(data_base, 'cv{}_train.csv'.format(i))
		val_path = os.path.join(data_base, 'cv{}_val.csv'.format(i))
		test_path = os.path.join(data_base, 'cv{}_test.csv'.format(i))
		cv_path_ls.append([train_path, val_path, test_path])

	if model_name in ['dkt', 'DKT']:
		for i in range(5):
			ckpt_path_ls.append(os.path.join(ckpt_base, 'pt_cv{}.pt'.format(i)))
	elif model_name in ['ode_dkt', 'ode_DKT']:
		for i in range(5):
			ckpt_path_ls.append(os.path.join(ckpt_base, 'ft_cv{}.pt'.format(i)))
	else:
		raise Exception('model name is wrong, current model_name:{}'.format(model_name))
	
	return cv_path_ls, ckpt_path_ls





def preview_dataloader_new(dataloader_ls):
	train_dataloader, val_dataloader, test_dataloader = dataloader_ls[0], dataloader_ls[1], dataloader_ls[2]
	for i, item in enumerate(train_dataloader):
		print('train skill_rsps_hots shape: ', item['skill_rsps_hots'].shape)
		print('train skill_hots shape: ', item['skill_hots'].shape)
		print('train rsps_array shape: ', item['rsps_array'].shape)
		print('train skill_array shape: ', item['skill_array'].shape)
		print('train timesteps shape: ', item['timesteps'].shape)
		print('==============================================================')
		break
	for i, item in enumerate(val_dataloader):
		print('val batch shape: ', item['skill_rsps_hots'].shape)
		break
	for i, item in enumerate(test_dataloader):
		print('test batch shape: ', item['skill_rsps_hots'].shape)
		break
	return



def preview_dataloader(dataloader_ls, mode = 'train'):
	train_dataloader, val_dataloader, test_dataloader = dataloader_ls[0], dataloader_ls[1], dataloader_ls[2]
	if mode ==' train':
		for i, item in enumerate(train_dataloader):
			print('train batch shape: ', item[0].shape)
			break
		for i, item in enumerate(val_dataloader):
			print('val batch shape: ', item[0].shape)
			break
		for i, item in enumerate(test_dataloader):
			print('test batch shape: ', item[0].shape)
			break
	else:
		for i, item in enumerate(test_dataloader):
			print('test batch shape: ', item[0].shape)
			break





def scaled_dot_product_attention_shuffle(q, k, v, mask):
	"""Calculate the attention weights.
	q, k, v must have matching leading dimensions.
	k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
	The mask has different shapes depending on its type(padding or look ahead)
	but it must be broadcastable for addition.

	Args:
		q: query shape == (..., seq_len_q, depth)
		k: key shape == (..., seq_len_k, depth)
		v: value shape == (..., seq_len_v, depth_v)
		mask: Float tensor with shape broadcastable
					to (..., seq_len_q, seq_len_k). Defaults to None.

	Returns:
		output, attention_weights
	"""
	# k: (bs, sl, n_head, sl, depth)
	sl, dk = k.size(-2), k.size(-1)
	matmul_qk = torch.matmul(q, k.permute(0,1,2,4,3)) # (bs, sl, n_head, sl_q, sl_k)
	
	# scale matmul_qk
	dk = torch.tensor(dk, dtype = torch.float32)
	scaled_attention_logits = matmul_qk / torch.sqrt(dk) # (bs, sl, n_head, sl_q, sl_k)
	# add the mask to the scaled tensor.
	if mask is not None:
		scaled_attention_logits += (mask.to(q.device) * -1e9)  

	# softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
	attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # # (bs, sl, n_head, sl_q, sl_k)
	# attention_weights or att_score: 
	#    (bs, sl, n_head, sl_q, sl_k)
	# v: (bs, sl, n_head, sl_v, depth): torch.Size([128, 2, 99, 64])
	output = torch.matmul(attention_weights, v) # (bs, sl, n_head, sl_q, depth)
	return output, attention_weights


# compute_sigmoid_promotion
def compute_minmax_promotion_linear(x): 
	# att_score: (batch_size, num_heads, seq_len_q, seq_len_k)\
	sl = x.size(-1)
	promote_matrix = torch.zeros(sl, sl).to(x.device)
	for i in range(sl):
		promote_matrix[i][:i+1] = torch.arange(1,i+2,1)/(i+1)
	x *= promote_matrix
	return x



def compute_minmax_promotion(x): 
	# att_score: (batch_size, num_heads, seq_len_q, seq_len_k)\
	sl = x.size(-1)
	promote_matrix = torch.zeros(sl, sl).to(x.device)
	for i in range(sl):
		promote_matrix[i][:i+1] = torch.exp(torch.arange(1,i+2,1)/(i+1))
	x *= promote_matrix
	return x

# promote_matrix = compute_minmax_promotion(x = torch.rand(99,99))
# print(promote_matrix[98])


def scaled_dot_product_attention(q, k, v, mask):
	sl, dk = k.size(-2), k.size(-1)
	# print('q, k')
	# print(q[0,0,:,:5])
	# print(k.permute(0,1,3,2)[0,0,:5,:])
	matmul_qk = torch.matmul(q, k.permute(0,1,3,2)) # -> q,k,v: (batch_size, num_heads, depth, seq_len_q)
	# print('matmul_qk')
	# print(matmul_qk)
	# scale matmul_qk
	dk = torch.tensor(dk, dtype = torch.float32)
	scaled_attention_logits = matmul_qk / torch.sqrt(dk) # (batch_size, num_heads, seq_len_q, seq_len_k)

	# print('scaled_attention_logits')
	# print(scaled_attention_logits)
	# add the mask to the scaled tensor.
	if mask is not None:
		scaled_attention_logits += (mask.to(q.device) * -1e9)  

	# softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
	attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # (..., seq_len_q, seq_len_k)
	# attention_weights or att_score: 
	#    (batch_size, num_heads, seq_len_q, seq_len_k)
	# v: (batch_size, num_heads, seq_len_v, depth): torch.Size([128, 2, 99, 64])
	output = torch.matmul(attention_weights, v) 
	return output, attention_weights # att_weights: (batch_size, num_heads, seq_len_q, seq_len_k)



def scaled_dot_product_attention_promote(q, k, v, mask):
	sl, dk = k.size(-2), k.size(-1)
	matmul_qk = torch.matmul(q, k.permute(0,1,3,2)) # -> q,k,v: (batch_size, num_heads, depth, seq_len_q)
	
	# scale matmul_qk
	dk = torch.tensor(dk, dtype = torch.float32)
	scaled_attention_logits = matmul_qk / torch.sqrt(dk) # (batch_size, num_heads, seq_len_q, seq_len_k)

	scaled_attention_logits = compute_minmax_promotion(scaled_attention_logits)
	# add the mask to the scaled tensor.
	if mask is not None:
		scaled_attention_logits += (mask.to(q.device) * -1e9)  

	# softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
	attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # (..., seq_len_q, seq_len_k)
	# attention_weights or att_score: 
	#    (batch_size, num_heads, seq_len_q, seq_len_k)
	# v: (batch_size, num_heads, seq_len_v, depth): torch.Size([128, 2, 99, 64])
	output = torch.matmul(attention_weights, v) 
	return output, attention_weights # att_weights: (batch_size, num_heads, seq_len_q, seq_len_k)



# def compute_expnential_decay(x, b): 
# 	# y = x * b^t
# 	# att_score: (batch_size, num_heads, seq_len_q, seq_len_k)\
# 	sl = x.size(-1)
# 	decay_matrix = torch.zeros(sl, sl).to(x.device)
# 	for i in range(sl):
# 		decay_matrix[i][:i+1] = torch.arange(i,0,-1)
# 	decay_matrix = torch.exp(-decay_matrix)
# 	return


class MultiHeadAttention(nn.Module):
	def __init__(self, input_dim, d_model, num_heads, dropout = 0):
		super(MultiHeadAttention, self).__init__()
		self.num_heads = num_heads
		self.d_model = d_model

		assert d_model % self.num_heads == 0

		self.depth = d_model // self.num_heads

		self.wq = nn.Linear(input_dim, d_model)
		self.wk = nn.Linear(input_dim, d_model)
		self.wv = nn.Linear(input_dim, d_model)
		
		self.dense = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(dropout)

	def split_heads(self, x):
		"""Split the last dimension into (num_heads, depth).
		Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
		"""
		bs, sl, d_ = x.size()
		x = torch.reshape(x, (bs, sl, self.num_heads, self.depth)) # (5, 10, 10, 2, 8)
		return x.permute(0, 2, 1, 3)

	def forward(self, q, k, v, mask):
		bs, sl, d_ = q.size()

		q = self.wq(q)	# (batch_size, seq_len, d_model)
		k = self.wk(k)	# (batch_size, seq_len, d_model)
		v = self.wv(v)	# (batch_size, seq_len, d_model)

		print('q, k, v')
		print(q[0:,:,:5])
		print(k[0:,:,:5])
		print(v[0:,:,:5])

		q = self.split_heads(q)	# (batch_size, num_heads, seq_len_q, depth)
		k = self.split_heads(k)	# (batch_size, num_heads, seq_len_k, depth)
		v = self.split_heads(v)	# (batch_size, num_heads, seq_len_v, depth)

		# print('q, k, v')
		# print(q[0,0,:,:5])
		# print(k[0,0,:,:5])
		# print(v[0,0,:,:5])

		# scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
		# attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
		scaled_attention, attention_weights = scaled_dot_product_attention(
				q, k, v, mask)
		
		# print('attn weights')
		# print(attention_weights)

		scaled_attention = scaled_attention.permute(0, 2, 1, 3) # -> (batch_size, seq_len_q, num_heads, depth)
		concat_attention = torch.reshape(scaled_attention, (bs, -1, self.d_model))
		assert concat_attention.size(1) == sl

		output = self.dense(concat_attention)	# (batch_size, seq_len_q, d_model)

		# print('attention activation')
		# print(output[0:,:,:5])

		output = self.dropout(output)
		# print('dropout activation')
		# print(output[0:,:,:5])

		return output, attention_weights




class MultiHeadAttentionShuffle(nn.Module):
	def __init__(self, input_dim, d_model, num_heads, dropout = 0):
		super(MultiHeadAttentionShuffle, self).__init__()
		self.num_heads = num_heads
		self.d_model = d_model

		assert d_model % self.num_heads == 0

		self.depth = d_model // self.num_heads

		self.wq = nn.Linear(input_dim, d_model)
		self.wk = nn.Linear(input_dim, d_model)
		self.wv = nn.Linear(input_dim, d_model)
		
		self.dense = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(dropout)

	def copy_split_heads(self, x):
		"""Split the last dimension into (num_heads, depth).
		Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
		"""
		bs, sl, d_ = x.size()
		x_t_ = torch.zeros(bs,sl,sl,d_).to(x.device)
		for t in range(sl):
			shuffled = cat((x[:,:t,:][:,randperm(t),:], x[:,t:t+1,:]), dim=1)
			# print('shuffled.size()', shuffled.size())
			plc_holder = torch.zeros(bs, sl-(t+1), d_).to(x.device)
			# print('plc_holder.size()', plc_holder.size())
			x_t_[:,t,:,:] = cat((shuffled, plc_holder), dim = 1) # x_t_ is the steps before(include) t, (bs, sl, d_)
		x_t_ = torch.reshape(x_t_, (bs, sl, sl, self.num_heads, self.depth)) # (5, 10, 10, 2, 8)
		return x_t_.permute(0, 1, 3, 2, 4) # (0,2,1,3)

	def forward(self, q, k, v, mask):
		bs, sl, d_ = q.size()

		q = self.wq(q)	# (bs, sl, d_)
		k = self.wk(k)	
		v = self.wv(v)	

		q = self.copy_split_heads(q)	# (bs, sl, n_head, sl, depth)
		k = self.copy_split_heads(k)	
		v = self.copy_split_heads(v)	

		# scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
		# attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
		scaled_attention, attention_weights = scaled_dot_product_attention_shuffle(
				q, k, v, mask) # (bs, sl, n_head, sl_q, depth)

		scaled_attention = scaled_attention.permute(0, 1, 3, 2, 4) # (bs, sl, n_head, sl_q, depth) -> (bs, sl, sl_q, n_head, depth)
		concat_attention = torch.reshape(scaled_attention, (bs, sl, -1, self.d_model))
		assert concat_attention.size(1) == sl

		idx_dim1 = torch.arange(0,sl)
		idx_dim2 = torch.arange(0,sl)
		effective_output = concat_attention[:,idx_dim1,idx_dim2,:]
		output = self.dense(effective_output) # (batch_size, seq_len_q, d_model)
		output = self.dropout(output)

		return output, attention_weights



class MultiHeadAttentionPromotion(nn.Module):
	def __init__(self, input_dim, d_model, num_heads, dropout = 0):
		super(MultiHeadAttentionPromotion, self).__init__()
		self.num_heads = num_heads
		self.d_model = d_model

		assert d_model % self.num_heads == 0

		self.depth = d_model // self.num_heads

		self.wq = nn.Linear(input_dim, d_model)
		self.wk = nn.Linear(input_dim, d_model)
		self.wv = nn.Linear(input_dim, d_model)
		
		self.dense = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(dropout)

	def split_heads(self, x):
		"""Split the last dimension into (num_heads, depth).
		Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
		"""
		bs, sl, d_ = x.size()
		x = torch.reshape(x, (bs, sl, self.num_heads, self.depth)) # (5, 10, 10, 2, 8)
		return x.permute(0, 2, 1, 3)

	def forward(self, q, k, v, mask):
		bs, sl, d_ = q.size()

		q = self.wq(q)	# (batch_size, seq_len, d_model)
		k = self.wk(k)	# (batch_size, seq_len, d_model)
		v = self.wv(v)	# (batch_size, seq_len, d_model)

		q = self.split_heads(q)	# (batch_size, num_heads, seq_len_q, depth)
		k = self.split_heads(k)	# (batch_size, num_heads, seq_len_k, depth)
		v = self.split_heads(v)	# (batch_size, num_heads, seq_len_v, depth)

		# scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
		# attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
		scaled_attention, attention_weights = scaled_dot_product_attention_promote(
				q, k, v, mask)

		scaled_attention = scaled_attention.permute(0, 2, 1, 3) # -> (batch_size, seq_len_q, num_heads, depth)
		concat_attention = torch.reshape(scaled_attention, (bs, -1, self.d_model))
		assert concat_attention.size(1) == sl

		output = self.dense(concat_attention)	# (batch_size, seq_len_q, d_model)
		output = self.dropout(output)

		return output, attention_weights