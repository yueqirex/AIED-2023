import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from dkt_load_data import DktDataloader
import dkt_load_data
from torch import cat, eye, gather, randperm

import numpy as np
import pandas as pd

import sys


import math
from importlib_metadata import requires
import torch
import torch.nn as nn
from torch import cat, eye, gather, randperm
import numpy as np
import time

'''

lstm1 = IdenLSTM(2000,2000,1,0.1)
# lstm1.to('cuda:6')
lstm2 = IdenShflLSTM(2000,2000,1,0.1)
# lstm2.to('cuda:6')
hidden_seq, states = lstm1(torch.rand(64,100,2000))
hidden_seq, states = lstm2(torch.rand(64,100,2000))

GPU
--- IdenLSTM 0.3617277145385742 seconds ---
--- data parallel time 2.1067943572998047 seconds ---
--- data process time 4.173916816711426 seconds ---
--- IdenShflLSTM 10.942209482192993 seconds --

CPU
--- IdenLSTM 2.1373207569122314 seconds ---
--- data parallel time 2.9470651149749756 seconds ---
--- data process time 109.37675380706787 seconds ---
--- IdenShflLSTM 112.71035623550415 seconds ---
'''

class RegLSTM(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers = 1, dropout = 0.0):
		super(RegLSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		W_param_ls, U_param_ls, b_param_ls = [], [], []
		for i in range(num_layers):
			if i==0:
				W_param_ls.append(nn.Parameter(torch.Tensor(input_size, hidden_size * 4)))
				U_param_ls.append(nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4)))
				b_param_ls.append(nn.Parameter(torch.Tensor(hidden_size * 4))) # W, U, b for 1st layer
			else:
				W_param_ls.append(nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4)))
				U_param_ls.append(nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4)))
				b_param_ls.append(nn.Parameter(torch.Tensor(hidden_size * 4))) # W, U, b for subsequent layer
		self.W_param_ls = nn.ParameterList(W_param_ls)
		self.U_param_ls = nn.ParameterList(U_param_ls)
		self.b_param_ls = nn.ParameterList(b_param_ls)
		self.dropout = nn.Dropout(dropout)
		self.init_weights()
		
	def init_weights(self):
		stdv = 1.0 / math.sqrt(self.hidden_size)
		for weight in self.parameters():
			weight.data.uniform_(-stdv, stdv)
		 
	def forward(self, x, init_states=None):
		# hidden_seq, h_t, c_t = None, None, None
		# start_time = time.time()
		for i in range(self.num_layers):
			"""Assumes x is of shape (batch, sequence, feature)"""
			bs, seq_sz, _ = x.size()
			hidden_seq = torch.zeros(bs, seq_sz, self.hidden_size).to(x.device) # size: bs, seq_sz, feature
			if init_states is None:
				h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
							torch.zeros(bs, self.hidden_size).to(x.device))
			else:
				h_t, c_t = init_states
			HS = self.hidden_size
			# start_time1 = time.time()
			for t in range(seq_sz):
				x_t = x[:, t, :]
				# batch the computations into a single matrix multiplication
				gates = x_t @ self.W_param_ls[i] + h_t @ self.U_param_ls[i] + self.b_param_ls[i]
					
				i_t, f_t, g_t, o_t = (
					torch.sigmoid(gates[:, :HS]), # input
					torch.sigmoid(gates[:, HS:HS*2]), # forget
					torch.tanh(gates[:, HS*2:HS*3]),
					torch.sigmoid(gates[:, HS*3:]), # output
				)
				c_t = f_t * c_t + i_t * g_t
				h_t = o_t * torch.tanh(c_t)
				hidden_seq[:,t,:] = h_t
				# print('hidden_seq[:,t,:].size()', hidden_seq[:,t,:].size())
			# reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
			# if i != self.num_layers-1:
				# hidden_seq = self.dropout(hidden_seq)
			# print('--- data processing %s seconds ---' % (time.time()-start_time1))
			x = hidden_seq
		# print('--- IdenLSTM %s seconds ---' % (time.time()-start_time))
		return hidden_seq, (h_t, c_t)

class IdenLSTM(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers = 1, dropout = 0.0):
		super(IdenLSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		W_param_ls, U_param_ls, b_param_ls = [], [], []

		b = eye(self.hidden_size, requires_grad = False)
		U = nn.Parameter(cat((b,b,b,b),-1))
		for i in range(num_layers):
			if i==0:
				# W for 1st layer
				W_param_ls.append(nn.Parameter(torch.Tensor(input_size, hidden_size * 4)))
			else:
				 # W for subsequent layer
				W_param_ls.append(nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4)))
			U_param_ls.append(U)
			b_param_ls.append(nn.Parameter(torch.Tensor(hidden_size * 4)))
		self.W_param_ls = nn.ParameterList(W_param_ls)
		self.U_param_ls = nn.ParameterList(U_param_ls)
		self.b_param_ls = nn.ParameterList(b_param_ls)
		self.dropout = nn.Dropout(dropout)
		self.init_weights()
		
	def init_weights(self):
		stdv = 1.0 / math.sqrt(self.hidden_size)
		for name, weight in self.named_parameters():
			if "U_param_ls" in name:
				weight.requires_grad = False
			else:
				weight.data.uniform_(-stdv, stdv)
		 
	def forward(self, x, init_states=None):
		# hidden_seq, h_t, c_t = None, None, None
		# start_time = time.time()
		for i in range(self.num_layers):
			"""Assumes x is of shape (batch, sequence, feature)"""
			bs, seq_sz, _ = x.size()
			hidden_seq = torch.zeros(bs, seq_sz, self.hidden_size).to(x.device) # size: bs, seq_sz, feature
			if init_states is None:
				h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
							torch.zeros(bs, self.hidden_size).to(x.device))
			else:
				h_t, c_t = init_states
			HS = self.hidden_size
			# start_time1 = time.time()
			for t in range(seq_sz):
				x_t = x[:, t, :]
				# batch the computations into a single matrix multiplication
				gates = x_t @ self.W_param_ls[i] + h_t @ self.U_param_ls[i] + self.b_param_ls[i]
					
				i_t, f_t, g_t, o_t = (
					torch.sigmoid(gates[:, :HS]), # input
					torch.sigmoid(gates[:, HS:HS*2]), # forget
					torch.tanh(gates[:, HS*2:HS*3]),
					torch.sigmoid(gates[:, HS*3:]), # output
				)
				c_t = f_t * c_t + i_t * g_t
				h_t = o_t * torch.tanh(c_t)
				hidden_seq[:,t,:] = h_t
				# print('hidden_seq[:,t,:].size()', hidden_seq[:,t,:].size())
			# reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
			# if i != self.num_layers-1:
				# hidden_seq = self.dropout(hidden_seq)
			# print('--- data processing %s seconds ---' % (time.time()-start_time1))
			x = hidden_seq
		# print('--- IdenLSTM %s seconds ---' % (time.time()-start_time))
		return hidden_seq, (h_t, c_t)

class FixLSTM(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers = 1, dropout = 0.0):
		super(FixLSTM, self).__init__()
		self.input_size = input_size
		# self.hidden_size = hidden_size
		self.hidden_size = input_size
		self.num_layers = num_layers
		
		a, b = eye(self.input_size, requires_grad = False), eye(self.hidden_size, requires_grad = False)
		self.W_0 = nn.Parameter(cat((a,a,a,a),-1))
		self.W = nn.Parameter(cat((b,b,b,b),-1))
		self.U = nn.Parameter(cat((b,b,b,b),-1))
		# self.bias = nn.Parameter(eye((num_layers * hidden_size * 4), requires_grad = False))
		# self.dropout = nn.Dropout(dropout)
		self.init_weights()
		
	def init_weights(self):
		# stdv = 1.0 / math.sqrt(self.hidden_size)
		for weight in self.parameters():
			# weight.data.uniform_(-stdv, stdv)
			weight.requires_grad = False
			# print('===================================')
			# print(weight.data)
		 
	def forward(self, x, init_states=None):
		# hidden_seq, h_t, c_t = None, None, None
		# start_time = time.time()
		for i in range(self.num_layers):
			"""Assumes x is of shape (batch, sequence, feature)"""
			bs, seq_sz, _ = x.size()
			hidden_seq = torch.zeros(bs, seq_sz, self.hidden_size).to(x.device) # size: bs, seq_sz, feature
			if init_states is None:
				h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
							torch.zeros(bs, self.hidden_size).to(x.device))
			else:
				h_t, c_t = init_states
			HS = self.hidden_size
			# start_time1 = time.time()
			for t in range(seq_sz):
				x_t = x[:, t, :]
				# batch the computations into a single matrix multiplication
				if i == 0:
					gates = x_t @ self.W_0 + h_t @ self.U
				else:
					# print('=====================')
					# print(x_t.shape, self.W.shape, h_t.shape, self.U.shape)
					gates = x_t @ self.W + h_t @ self.U
					
				i_t, f_t, g_t, o_t = (
					torch.sigmoid(gates[:, :HS]), # input
					torch.sigmoid(gates[:, HS:HS*2]), # forget
					torch.tanh(gates[:, HS*2:HS*3]),
					torch.sigmoid(gates[:, HS*3:]), # output
				)
				c_t = f_t * c_t + i_t * g_t
				h_t = o_t * torch.tanh(c_t)
				hidden_seq[:,t,:] = h_t
				# print('hidden_seq[:,t,:].size()', hidden_seq[:,t,:].size())
			# reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
			# if i != self.num_layers-1:
				# hidden_seq = self.dropout(hidden_seq)
			# print('--- data processing %s seconds ---' % (time.time()-start_time1))
			x = hidden_seq
		# print('--- IdenLSTM %s seconds ---' % (time.time()-start_time))
		return hidden_seq, (h_t, c_t)

class ShflLSTM(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers = 1, dropout = 0.0):
		super(ShflLSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		W_param_ls, U_param_ls, b_param_ls = [], [], []
		for i in range(num_layers):
			if i==0:
				W_param_ls.append(nn.Parameter(torch.Tensor(input_size, hidden_size * 4)))
				U_param_ls.append(nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4)))
				b_param_ls.append(nn.Parameter(torch.Tensor(hidden_size * 4))) # W, U, b for 1st layer
			else:
				W_param_ls.append(nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4)))
				U_param_ls.append(nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4)))
				b_param_ls.append(nn.Parameter(torch.Tensor(hidden_size * 4))) # W, U, b for subsequent layer
		self.W_param_ls = nn.ParameterList(W_param_ls)
		self.U_param_ls = nn.ParameterList(U_param_ls)
		self.b_param_ls = nn.ParameterList(b_param_ls)
		self.dropout = nn.Dropout(dropout)
		self.init_weights()
		
	def init_weights(self):
		stdv = 1.0 / math.sqrt(self.hidden_size)
		for weight in self.parameters():
			weight.data.uniform_(-stdv, stdv)
		 
	def forward(self, x, init_states=None):
		# hidden_seq, h_t, c_t = None, None, None
		# start_time = time.time()
		for i in range(self.num_layers):
			"""Assumes x is of shape (batch, sequence, feature)"""
			bs, seq_sz, d_ = x.size()
			hidden_seq = torch.zeros(bs, seq_sz, self.hidden_size).to(x.device) # size: bs, seq_sz, feature
			if init_states is None:
				h_t, c_t = (torch.zeros(bs, seq_sz, self.hidden_size).to(x.device),
							torch.zeros(bs, seq_sz, self.hidden_size).to(x.device))
			else:
				h_t, c_t = init_states
			HS = self.hidden_size
			
			x_t_ = torch.zeros(bs,seq_sz,seq_sz,d_).to(x.device)
			# start_time1 = time.time()
			for t in range(seq_sz):    
				shuffled = cat((x[:,:t,:][:,randperm(t),:], x[:,t:t+1,:]), dim=1)
				# print('shuffled.size()', shuffled.size())
				plc_holder = torch.zeros(bs, seq_sz-(t+1), d_).to(x.device)
				# print('plc_holder.size()', plc_holder.size())
				x_t_[:,:,t,:] = cat((shuffled, plc_holder), dim = 1) # x_t_ is the steps before(include) t
			
			# start_time1 = time.time()
			for t in range(seq_sz):
				x_t = x_t_[:,t,:,:] # (bs,sl,d_)
				# batch the computations into a single matrix multiplication
				gates = x_t @ self.W_param_ls[i] + h_t @ self.U_param_ls[i] + self.b_param_ls[i]
					
				i_t, f_t, g_t, o_t = (
					torch.sigmoid(gates[:,:,:HS]), # update (bs,sl,d_)
					torch.sigmoid(gates[:,:,HS:HS*2]), # forget (bs,sl,d_)
					torch.tanh(gates[:,:,HS*2:HS*3]), # c~ (bs,sl,d_)
					torch.sigmoid(gates[:,:,HS*3:]), # output (bs,sl,d_)
				)
				c_t = f_t * c_t + i_t * g_t
				h_t = o_t * torch.tanh(c_t) # (bs,sl,d_)
				
				hidden_seq[:,t,:] = h_t[:,t,:] # (bs,d_)
			x = hidden_seq
		# print('--- IdenShflLSTM %s seconds ---' % (time.time()-start_time))
		return hidden_seq, (h_t, c_t)

class IdenShflLSTM(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers = 1, dropout = 0.0):
		super(IdenShflLSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		W_param_ls, U_param_ls, b_param_ls = [], [], []

		b = eye(self.hidden_size, requires_grad = False)
		U = nn.Parameter(cat((b,b,b,b),-1))
		for i in range(num_layers):
			if i==0:
				# W for 1st layer
				W_param_ls.append(nn.Parameter(torch.Tensor(input_size, hidden_size * 4)))
			else:
				 # W for subsequent layer
				W_param_ls.append(nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4)))
			U_param_ls.append(U)
			b_param_ls.append(nn.Parameter(torch.Tensor(hidden_size * 4)))
		self.W_param_ls = nn.ParameterList(W_param_ls)
		self.U_param_ls = nn.ParameterList(U_param_ls)
		self.b_param_ls = nn.ParameterList(b_param_ls)
		self.dropout = nn.Dropout(dropout)
		self.init_weights()
		
	def init_weights(self):
		stdv = 1.0 / math.sqrt(self.hidden_size)
		for name, weight in self.named_parameters():
			if "U_param_ls" in name:
				weight.requires_grad = False
			else:
				weight.data.uniform_(-stdv, stdv)
		 
	def forward(self, x, init_states=None):
		# hidden_seq, h_t, c_t = None, None, None
		# start_time = time.time()
		for i in range(self.num_layers):
			"""Assumes x is of shape (batch, sequence, feature)"""
			bs, seq_sz, d_ = x.size()
			hidden_seq = torch.zeros(bs, seq_sz, self.hidden_size).to(x.device) # size: bs, seq_sz, feature
			if init_states is None:
				h_t, c_t = (torch.zeros(bs, seq_sz, self.hidden_size).to(x.device),
							torch.zeros(bs, seq_sz, self.hidden_size).to(x.device))
			else:
				h_t, c_t = init_states
			HS = self.hidden_size
			
			x_t_ = torch.zeros(bs,seq_sz,seq_sz,d_).to(x.device)
			# start_time1 = time.time()
			for t in range(seq_sz):    
				shuffled = cat((x[:,:t,:][:,randperm(t),:], x[:,t:t+1,:]), dim=1)
				# print('shuffled.size()', shuffled.size())
				plc_holder = torch.zeros(bs, seq_sz-(t+1), d_).to(x.device)
				# print('plc_holder.size()', plc_holder.size())
				x_t_[:,:,t,:] = cat((shuffled, plc_holder), dim = 1) # x_t_ is the steps before(include) t
			
			# start_time1 = time.time()
			for t in range(seq_sz):
				x_t = x_t_[:,t,:,:] # (bs,sl,d_)
				# batch the computations into a single matrix multiplication
				gates = x_t @ self.W_param_ls[i] + h_t @ self.U_param_ls[i] + self.b_param_ls[i]
					
				i_t, f_t, g_t, o_t = (
					torch.sigmoid(gates[:,:,:HS]), # update (bs,sl,d_)
					torch.sigmoid(gates[:,:,HS:HS*2]), # forget (bs,sl,d_)
					torch.tanh(gates[:,:,HS*2:HS*3]), # c~ (bs,sl,d_)
					torch.sigmoid(gates[:,:,HS*3:]), # output (bs,sl,d_)
				)
				c_t = f_t * c_t + i_t * g_t
				h_t = o_t * torch.tanh(c_t) # (bs,sl,d_)
				
				hidden_seq[:,t,:] = h_t[:,t,:] # (bs,d_)
			x = hidden_seq
		# print('--- IdenShflLSTM %s seconds ---' % (time.time()-start_time))
		return hidden_seq, (h_t, c_t)


# ============ custom DKT ==============
def reduce_dim(x):
  x = K.max(x, axis = -1, keepdims = True)
  return x

def one_hot(skill_matrix, vocab_size): # vocab_size is the total # of skills
  seq_len = skill_matrix.shape[1] # The length of a single record = the number of questions chosen = 100
  result = np.zeros((skill_matrix.shape[0], seq_len, vocab_size))
  for i in range(skill_matrix.shape[0]): # For every student:
    result[i, np.arange(seq_len), skill_matrix[i]] = 1. 
  return result

def dkt_one_hot(skill_matrix, response_matrix, vocab_size):
  seq_len = skill_matrix.shape[1] # the number of 100 questions answered.
  skill_response_array = np.zeros((skill_matrix.shape[0], seq_len, 2 * vocab_size)) # (2 * # of skills)
  for i in range(skill_matrix.shape[0]):
    skill_response_array[i, np.arange(seq_len), 2 * skill_matrix[i] + response_matrix[i]] = 1. # mark the response in the [vocab_size: 2*vocab_size] subspace.
  return skill_response_array

class RegDKT(nn.Module):
  def __init__(self, num_lstm_blocks = 1, input_dim = None, lstm_dim_ls = None, num_layers_ls = None, dropout_ls = None, device = None):
    super(RegDKT, self).__init__()
    self.num_lstm_blocks = num_lstm_blocks
    self.lstm_blocks = []
    for i in range(self.num_lstm_blocks):
      if i == 0:
        unit = RegLSTM(input_size = input_dim, hidden_size = input_dim, num_layers = num_layers_ls[i], dropout = dropout_ls[i])
        # unit.flatten_parameters()
        self.lstm_blocks.append(unit)
      else:
        unit = RegLSTM(input_size = input_dim, hidden_size = input_dim, num_layers = num_layers_ls[i], dropout = dropout_ls[i])
        # unit.flatten_parameters()  
        self.lstm_blocks.append(unit)
    self.lstm_box = nn.ModuleList(self.lstm_blocks)
    self.lr = nn.Linear(in_features = input_dim, out_features = int(input_dim/2))
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, prior_skill_rsps, post_skill): # one_hot input
    for i, lstm_i in enumerate(self.lstm_box):
      if i == 0:
        # lstm_i.flatten_parameters()
        lstm_h, states = lstm_i(prior_skill_rsps) # the last dimension would be n_skill in dkt or n_problem in q_dkt
      else:
        # lstm_i.flatten_parameters()
        lstm_h, states = lstm_i(lstm_h)
    lr_h = self.lr(lstm_h)
    one_hot_p = torch.mul(lr_h, post_skill)
    # print((one_hot_p>0).float().mean())
    # print('one_hot_p shape', one_hot_p.shape, one_hot_p)
    
    _, indices = torch.max(torch.abs(one_hot_p), dim = -1)
    indices = indices.unsqueeze(-1)
    pred = torch.gather(one_hot_p, -1, indices)
    pred = torch.squeeze(pred, -1)
    
    # print('pred shape', pred.shape, pred)
    pred = self.sigmoid(pred)
    return pred

class ShflDktCompatible(nn.Module):
  def __init__(self, num_lstm_blocks = 1, input_dim = None, lstm_dim_ls = None, num_layers_ls = None, dropout_ls = None, device = None):
    super(ShflDktCompatible, self).__init__()
    self.num_lstm_blocks = num_lstm_blocks
    self.lstm_blocks = []
    for i in range(self.num_lstm_blocks):
      if i == 0:
        unit = ShflLSTM(input_size = input_dim, hidden_size = input_dim, num_layers = num_layers_ls[i], dropout = dropout_ls[i])
        # unit.flatten_parameters()
        self.lstm_blocks.append(unit)
      else:
        unit = ShflLSTM(input_size = input_dim, hidden_size = input_dim, num_layers = num_layers_ls[i], dropout = dropout_ls[i])
        # unit.flatten_parameters()  
        self.lstm_blocks.append(unit)
    self.lstm_box = nn.ModuleList(self.lstm_blocks)
    self.lr = nn.Linear(in_features = input_dim, out_features = int(input_dim/2))
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, prior_skill_rsps, post_skill): # one_hot input
    for i, lstm_i in enumerate(self.lstm_box):
      if i == 0:
        # lstm_i.flatten_parameters()
        lstm_h, states = lstm_i(prior_skill_rsps) # the last dimension would be n_skill in dkt or n_problem in q_dkt
      else:
        # lstm_i.flatten_parameters()
        lstm_h, states = lstm_i(lstm_h)
    lr_h = self.lr(lstm_h)
    one_hot_p = torch.mul(lr_h, post_skill)
    # print((one_hot_p>0).float().mean())
    # print('one_hot_p shape', one_hot_p.shape, one_hot_p)
    
    _, indices = torch.max(torch.abs(one_hot_p), dim = -1)
    indices = indices.unsqueeze(-1)
    pred = torch.gather(one_hot_p, -1, indices)
    pred = torch.squeeze(pred, -1)
    
    # print('pred shape', pred.shape, pred)
    pred = self.sigmoid(pred)
    return pred

class IdenDKT(nn.Module):
  def __init__(self, num_lstm_blocks = 1, input_dim = None, lstm_dim_ls = None, num_layers_ls = None, dropout_ls = None, device = None):
    super(IdenDKT, self).__init__()
    self.num_lstm_blocks = num_lstm_blocks
    self.lstm_blocks = []
    for i in range(self.num_lstm_blocks):
      if i == 0:
        unit = IdenLSTM(input_size = input_dim, hidden_size = input_dim, num_layers = num_layers_ls[i], dropout = dropout_ls[i])
        # unit.flatten_parameters()
        self.lstm_blocks.append(unit)
      else:
        unit = IdenLSTM(input_size = input_dim, hidden_size = input_dim, num_layers = num_layers_ls[i], dropout = dropout_ls[i])
        # unit.flatten_parameters()  
        self.lstm_blocks.append(unit)
    self.lstm_box = nn.ModuleList(self.lstm_blocks)
    self.lr = nn.Linear(in_features = input_dim, out_features = int(input_dim/2))
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, prior_skill_rsps, post_skill): # one_hot input
    for i, lstm_i in enumerate(self.lstm_box):
      if i == 0:
        # lstm_i.flatten_parameters()
        lstm_h, states = lstm_i(prior_skill_rsps) # the last dimension would be n_skill in dkt or n_problem in q_dkt
      else:
        # lstm_i.flatten_parameters()
        lstm_h, states = lstm_i(lstm_h)
    lr_h = self.lr(lstm_h)
    one_hot_p = torch.mul(lr_h, post_skill)
    # print((one_hot_p>0).float().mean())
    # print('one_hot_p shape', one_hot_p.shape, one_hot_p)
    
    _, indices = torch.max(torch.abs(one_hot_p), dim = -1)
    indices = indices.unsqueeze(-1)
    pred = torch.gather(one_hot_p, -1, indices)
    pred = torch.squeeze(pred, -1)
    
    # print('pred shape', pred.shape, pred)
    pred = self.sigmoid(pred)
    return pred

class ShflDkt(nn.Module):
  def __init__(self, num_lstm_blocks = 1, input_dim = None, lstm_dim_ls = None, num_layers_ls = None, dropout_ls = None, device = None):
    super(ShflDkt, self).__init__()
    self.num_lstm_blocks = num_lstm_blocks
    self.lstm_blocks = []
    for i in range(self.num_lstm_blocks):
      if i == 0:
        unit = ShflLSTM(input_size = input_dim, hidden_size = lstm_dim_ls[i], num_layers = num_layers_ls[i], dropout = dropout_ls[i])
        # unit.flatten_parameters()
        self.lstm_blocks.append(unit)
      else:
        unit = ShflLSTM(input_size = lstm_dim_ls[i-1], hidden_size = lstm_dim_ls[i], num_layers = num_layers_ls[i], dropout = dropout_ls[i])
        # unit.flatten_parameters()  
        self.lstm_blocks.append(unit)
    self.lstm_box = nn.ModuleList(self.lstm_blocks)
    # self.lstm_box = self.lstm_blocks
    self.lr = nn.Linear(in_features = lstm_dim_ls[-1], out_features = int(input_dim/2))
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, prior_skill_rsps, post_skill): # one_hot input
    for i, lstm_i in enumerate(self.lstm_box):
      if i == 0:
        # lstm_i.flatten_parameters()
        lstm_h, states = lstm_i(prior_skill_rsps) # the last dimension would be n_skill in dkt or n_problem in q_dkt
      else:
        # lstm_i.flatten_parameters()
        lstm_h, states = lstm_i(lstm_h)
    lr_h = self.lr(lstm_h)
    # print(lr_h.size(), post_skill.size())
    one_hot_p = torch.mul(lr_h, post_skill)
    # print((one_hot_p>0).float().mean())
    # print('one_hot_p shape', one_hot_p.shape, one_hot_p)
    
    _, indices = torch.max(torch.abs(one_hot_p), dim = -1)
    indices = indices.unsqueeze(-1)
    pred = torch.gather(one_hot_p, -1, indices)
    pred = torch.squeeze(pred, -1)
    
    # print('pred shape', pred.shape, pred)
    pred = self.sigmoid(pred)
    return pred


class IdenShflDktCompatible(nn.Module):
  def __init__(self, num_lstm_blocks = 1, input_dim = None, lstm_dim_ls = None, num_layers_ls = None, dropout_ls = None, device = None):
    super(IdenShflDktCompatible, self).__init__()
    self.num_lstm_blocks = num_lstm_blocks
    self.lstm_blocks = []
    for i in range(self.num_lstm_blocks):
      if i == 0:
        unit = IdenShflLSTM(input_size = input_dim, hidden_size = input_dim, num_layers = num_layers_ls[i], dropout = dropout_ls[i])
        # unit.flatten_parameters()
        self.lstm_blocks.append(unit)
      else:
        unit = IdenShflLSTM(input_size = input_dim, hidden_size = input_dim, num_layers = num_layers_ls[i], dropout = dropout_ls[i])
        # unit.flatten_parameters()  
        self.lstm_blocks.append(unit)
    self.lstm_box = nn.ModuleList(self.lstm_blocks)
    self.lr = nn.Linear(in_features = input_dim, out_features = int(input_dim/2))
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, prior_skill_rsps, post_skill): # one_hot input
    for i, lstm_i in enumerate(self.lstm_box):
      if i == 0:
        # lstm_i.flatten_parameters()
        lstm_h, states = lstm_i(prior_skill_rsps) # the last dimension would be n_skill in dkt or n_problem in q_dkt
      else:
        # lstm_i.flatten_parameters()
        lstm_h, states = lstm_i(lstm_h)
    lr_h = self.lr(lstm_h)
    one_hot_p = torch.mul(lr_h, post_skill)
    # print((one_hot_p>0).float().mean())
    # print('one_hot_p shape', one_hot_p.shape, one_hot_p)
    
    _, indices = torch.max(torch.abs(one_hot_p), dim = -1)
    indices = indices.unsqueeze(-1)
    pred = torch.gather(one_hot_p, -1, indices)
    pred = torch.squeeze(pred, -1)
    
    # print('pred shape', pred.shape, pred)
    pred = self.sigmoid(pred)
    return pred


class IdenShflDkt(nn.Module):
  def __init__(self, num_lstm_blocks = 1, input_dim = None, lstm_dim_ls = None, num_layers_ls = None, dropout_ls = None, device = None):
    super(IdenShflDkt, self).__init__()
    self.num_lstm_blocks = num_lstm_blocks
    self.lstm_blocks = []
    for i in range(self.num_lstm_blocks):
      if i == 0:
        unit = IdenShflLSTM(input_size = input_dim, hidden_size = lstm_dim_ls[i], num_layers = num_layers_ls[i], dropout = dropout_ls[i])
        # unit.flatten_parameters()
        self.lstm_blocks.append(unit)
      else:
        unit = IdenShflLSTM(input_size = lstm_dim_ls[i-1], hidden_size = lstm_dim_ls[i], num_layers = num_layers_ls[i], dropout = dropout_ls[i])
        # unit.flatten_parameters()  
        self.lstm_blocks.append(unit)
    self.lstm_box = nn.ModuleList(self.lstm_blocks)
    # self.lstm_box = self.lstm_blocks
    self.lr = nn.Linear(in_features = lstm_dim_ls[-1], out_features = int(input_dim/2))
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, prior_skill_rsps, post_skill): # one_hot input
    for i, lstm_i in enumerate(self.lstm_box):
      if i == 0:
        # lstm_i.flatten_parameters()
        lstm_h, states = lstm_i(prior_skill_rsps) # the last dimension would be n_skill in dkt or n_problem in q_dkt
      else:
        # lstm_i.flatten_parameters()
        lstm_h, states = lstm_i(lstm_h)
    lr_h = self.lr(lstm_h)
    # print(lr_h.size(), post_skill.size())
    one_hot_p = torch.mul(lr_h, post_skill)
    # print((one_hot_p>0).float().mean())
    # print('one_hot_p shape', one_hot_p.shape, one_hot_p)
    
    _, indices = torch.max(torch.abs(one_hot_p), dim = -1)
    indices = indices.unsqueeze(-1)
    pred = torch.gather(one_hot_p, -1, indices)
    pred = torch.squeeze(pred, -1)
    
    # print('pred shape', pred.shape, pred)
    pred = self.sigmoid(pred)
    return pred

class FixDkt(nn.Module):
  def __init__(self, num_lstm_blocks = 1, input_dim = None, lstm_dim_ls = None, num_layers_ls = None, dropout_ls = None, device = None):
    super(FixDkt, self).__init__()
    self.num_lstm_blocks = num_lstm_blocks
    self.lstm_blocks = []
    for i in range(self.num_lstm_blocks):
      if i == 0:
        unit = IdenLSTM(input_size = input_dim, hidden_size = input_dim, num_layers = num_layers_ls[i], dropout = dropout_ls[i])
        # unit.flatten_parameters()
        self.lstm_blocks.append(unit)
      else:
        unit = IdenLSTM(input_size = input_dim, hidden_size = input_dim, num_layers = num_layers_ls[i], dropout = dropout_ls[i])
        # unit.flatten_parameters()  
        self.lstm_blocks.append(unit)
    self.lstm_box = nn.ModuleList(self.lstm_blocks)
    self.lr = nn.Linear(in_features = input_dim, out_features = int(input_dim/2))
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, prior_skill_rsps, post_skill): # one_hot input
    for i, lstm_i in enumerate(self.lstm_box):
      if i == 0:
        # lstm_i.flatten_parameters()
        lstm_h, states = lstm_i(prior_skill_rsps) # the last dimension would be n_skill in dkt or n_problem in q_dkt
      else:
        # lstm_i.flatten_parameters()
        lstm_h, states = lstm_i(lstm_h)
    lr_h = self.lr(lstm_h)
    one_hot_p = torch.mul(lr_h, post_skill)
    # print((one_hot_p>0).float().mean())
    # print('one_hot_p shape', one_hot_p.shape, one_hot_p)
    
    _, indices = torch.max(torch.abs(one_hot_p), dim = -1)
    indices = indices.unsqueeze(-1)
    pred = torch.gather(one_hot_p, -1, indices)
    pred = torch.squeeze(pred, -1)
    
    # print('pred shape', pred.shape, pred)
    pred = self.sigmoid(pred)
    return pred
    # return lstm_h[0].shape # ([32, 99, 64])
# train_path = './edm_2022/data/assist_09/sequential/dev0.csv'
# val_path = './edm_2022/data/assist_09/sequential/dev1.csv'
# test_path = './edm_2022/data/assist_09/sequential/dev2.csv'

# dkt_dl = DktDataloader(train_path, val_path, test_path)
# train_dataloader, val_dataloader, train_dataset, val_dataset = dkt_dl.get_data_loader()

# test_dkt = IdenShflDkt(input_dim = 20, lstm_dim_ls = [20], num_layers_ls=[1], dropout_ls = [0.0])

# for item in enumerate(train_datalaoder):
#   pred = test_dkt(item[0],item[1])  

# for item in train_dataloader:
#   pred = test_dkt(item[0][:,:-1].float(), item[1][:,1:].float())
#   print(pred.shape)
#   break