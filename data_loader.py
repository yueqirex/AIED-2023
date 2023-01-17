import sys
from torch.utils.data import Dataset, DataLoader
from dkt_bertkt_share.raw_load_data import PID_DATA_SPLIT
# from raw_load_data import PID_DATA
import numpy as np
import torch
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

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

'''
    Assistment 2009: n_skill = 110 # 123
'''
class KTDataset(Dataset):
	def __init__(self, n_question = None, max_seq = 100, separate_char = ',', path = None, model_name = None): # n_question: the # of skills
		super(KTDataset, self).__init__()
		assert model_name is not None
		self.max_seq = max_seq
		self.n_question = n_question
		self.path = path
		self.model_name = model_name
  
		pid_data = PID_DATA_SPLIT(n_question = n_question, seqlen = max_seq, separate_char = separate_char)
		self.p, self.q, self.qa, self.pure_qa, self.t, self.b = pid_data.load_data(path) 

		if 'dkt' in self.model_name or 'DKT' in self.model_name:
			self.skill_array = one_hot(self.q, n_question) # (rows, seq_len, n_skill)
			self.skill_response_array = dkt_one_hot(self.q, self.pure_qa, n_question) # (rows, seq_len, 2*n_skill)
    
	def __len__(self):
		return len(self.q) # rows = 950
  
	def __getitem__(self, index):
		# index is the internal interator index that should point to user_id
		# user_id = self.uid_list[index]
		if 'dkt' in self.model_name or 'DKT' in self.model_name:
			ski_array, skill_rsps_array, rsps_array, ski_index  = self.skill_array[index], self.skill_response_array[index], self.pure_qa[index], self.q[index]
			return skill_rsps_array, ski_array, rsps_array, ski_index
		elif 'bertkt' in self.model_name:
			p, q, qa, pure_qa, t, b = self.p[index], self.q[index], self.qa[index], self.pure_qa[index], self.t[index], self.b[index]
			return p, q, pure_qa, t, b


class KTDataloader():
	def __init__(self, train_path, val_path, test_path, n_question = None, batch_size = 2048, shuffle = False, num_workers = 0, model_name = None):
		self.train_path = train_path
		self.val_path = val_path
		self.test_path = test_path
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.num_workers = num_workers
		self.n_question = n_question
		self.model_name = model_name
		assert self.model_name is not None

	def get_data_loader(self):
		# train, val = train_test_split(group, test_size=0.2)
		train_dataset = KTDataset(path = self.train_path, n_question=self.n_question, model_name=self.model_name)
		train_dataloader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = self.shuffle, num_workers = self.num_workers, drop_last=False, worker_init_fn=seed_worker, generator=g)
		# del train
		val_dataset = KTDataset(path = self.val_path, n_question=self.n_question, model_name=self.model_name)
		val_dataloader = DataLoader(val_dataset, batch_size = self.batch_size, shuffle = False, num_workers = self.num_workers, drop_last=False, worker_init_fn=seed_worker, generator=g)

		test_dataset = KTDataset(path = self.test_path, n_question=self.n_question, model_name=self.model_name)
		test_dataloader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle = False, num_workers = self.num_workers, drop_last=False, worker_init_fn=seed_worker, generator=g)
		
		return train_dataloader, val_dataloader, test_dataloader, train_dataset, val_dataset, test_dataset