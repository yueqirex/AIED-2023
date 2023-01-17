import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import gc
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append('./custom_DKT/')
sys.path.append('./custom_AttnKT/')
sys.path.append('./')
import os
import os.path as path
import time
import json

import dkt_bertkt_share.data_loader as data_loader
import dkt_bertkt_share.train as train
import dkt_bertkt_share.utils as utils
from custom_DKT import IdenDKT

torch.manual_seed(0)
np.random.seed(0)

def train_proc(train_path, val_path, test_path, ckpt_path, batch_size = 256, n_question = None, model = None, epochs = None, shuffle = True,\
	num_workers = 8, hp = None, parallel = None, device_group = None):
	
	dkt_dataloader = dkt_ld.DktDataloader(train_path = train_path, val_path = val_path, test_path = test_path, n_question = n_question,\
											batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, data_type='short')
	train_dataloader, val_dataloader, test_dataloader, train_dataset, val_dataset, test_dataset = dkt_dataloader.get_data_loader()
	for i, item in enumerate(train_dataloader):
		print('train batch shape: ', item[0].shape)
		break
	for i, item in enumerate(val_dataloader):
		print('val batch shape: ', item[0].shape)
		break
	for i, item in enumerate(test_dataloader):
		print('test batch shape: ', item[0].shape)
		break

	# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=0.005)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	criterion = nn.BCELoss()

	'''==============================================Load_Model================================================='''
	# check_path = './nseq_DKT/ckpt/dkt_assistment_2009.pt'
	# if os.path.exists(check_path):
	#     model.load_state_dict(torch.load(check_path))
	#     model.eval()
	#     print('*******************Latest Weights Loaded to Model!*******************')

	'''=============================================Train_the_Model============================================='''
	# epochs = epochs
	# print(device)

	over_fit = 0
	last_auc, last_loss,last_rmse, last_combo = 0, 1e8, 1e8, 0
	ckpt_path = ckpt_path
	for epoch in tqdm(range(epochs)):
		# model, mode, data_loader, optim = None, criterion=None, device_group = None, parallel = False
		train_loss, train_acc, train_auc, train_rmse = train.epoch_parallel(model, 'train', train_dataloader, optimizer, criterion, device_group=device_group, parallel = parallel)
		if train_rmse == 0: train_rmse = 1e-8
		train_combo = (train_acc + train_auc + np.tanh([1.0/train_rmse])[0])/3.0
		print("epoch - {} train_loss - {:.3f} acc - {:.3f} auc - {:.3f} rmse - {:.3f} combo - {:.3f}".format(epoch+1, train_loss, train_acc, train_auc, train_rmse, train_combo))
		
		val_loss, val_acc, val_auc, val_rmse = train.epoch_parallel(model, 'val', val_dataloader, None, criterion, device_group=device_group, parallel = parallel)
		if val_rmse == 0: val_rmse = 1e-8
		val_combo = (val_acc + val_auc + np.tanh([1.0/val_rmse])[0])/3.0
		print("epoch - {} val_loss - {:.4f} acc - {:.4f} auc - {:.4f} rmse - {:.4f} combo - {:.4f}".format(epoch+1, val_loss, val_acc, val_auc, val_rmse, val_combo))

		test_loss, test_acc, test_auc, test_rmse, test_outs = train.epoch_parallel(model, 'test', test_dataloader, None, criterion, device_group=device_group, parallel = parallel)
		if test_rmse == 0: test_rmse = 1e-8
		test_combo = (test_acc + test_auc + np.tanh([1.0/test_rmse])[0])/3.0
		print("epoch - {} test_loss - {:.4f} acc - {:.4f} auc - {:.4f} rmse - {:.4f} combo - {:.4f}".format(epoch+1, test_loss, test_acc, test_auc, test_rmse, test_combo))
		'''=================================Save model ckpt================================='''
		# if epoch % 5 == 0:
		#     torch.save(model.state_dict(), ckpt_path)
		#     print('***NOTE***: Lastest model parameters saved to path!')

		'''=========================Use Loss as the early stop metric========================='''
		if np.round(val_loss, 4) < np.round(last_loss, 4):
			last_loss = val_loss
			over_fit = 0
			torch.save(model.state_dict(), ckpt_path)
			print('***NOTE***: Current best model parameters saved to path!')
		else:
			over_fit += 1
			print('***NOTE***: Model does not improve from the previous best')
		if over_fit >= 3: #2
			print("==============================================================")
			print("************NOTE: early stop at epoch: ", epoch, "************")
			break
	return epoch+1, test_loss, test_auc, test_acc, test_rmse, model, test_outs

def main():
	'''============================ 5-cv path ================================='''
	cv_path_ls = []
	for i in range(5):
		for j in range(3):
			if j == 0:
				train_path = './data/assist_09/sequential/cv{}_train.csv'.format(i)
			if j == 1:
				val_path = './data/assist_09/sequential/cv{}_val.csv'.format(i)
			if j == 2:
				test_path = './data/assist_09/sequential/cv{}_test.csv'.format(i)
				cv_path_ls.append([train_path, val_path, test_path])
				
	'''====================================== ckpt path ======================================'''
	ckpt_path1 = './nseq_DKT/cv_5_iden/assist_09_ckpt/cv1.pt' # ckpt for cv1 as validation set
	ckpt_path2 = './nseq_DKT/cv_5_iden/assist_09_ckpt/cv2.pt'
	ckpt_path3 = './nseq_DKT/cv_5_iden/assist_09_ckpt/cv3.pt'
	ckpt_path4 = './nseq_DKT/cv_5_iden/assist_09_ckpt/cv4.pt'
	ckpt_path5 = './nseq_DKT/cv_5_iden/assist_09_ckpt/cv5.pt'

	# ckpt_path_ls = [eval('ckpt_path{}'.format(i+1)) for i in range(5)]
	ckpt_path_ls = [ckpt_path1, ckpt_path2, ckpt_path3, ckpt_path4, ckpt_path5]

	'''============================Hyper-parameters=========================================='''
	# hp_best = {'num_layers':4, 'num_heads':8,'vocab_size':17751,'d_model':256,'max_seq':100}
	# hp_best = {"num_layers": 2, "num_heads": 4, "n_prob": 51569, "n_skill": 934, "d_model": 64}
	hp_best = {'num_lstm_blocks': 2, 'input_dim': 304, 'lstm_dim_ls':[128, 256], 'num_layers_ls': [1, 1], 'dropout_ls':[0.1, 0.1]}
	num_lstm_blocks, input_dim, lstm_dim_ls, num_layers_ls, dropout_ls =\
			hp_best['num_lstm_blocks'], hp_best['input_dim'], hp_best['lstm_dim_ls'], hp_best['num_layers_ls'], hp_best['dropout_ls']

	start_time = time.time()
	loss_ls, auc_ls, acc_ls, rmse_ls = [], [], [], []
	raw_pred_ls = [] # (5 * sample_num)
	for i in range(1):
		print("==============================================================")
		print("*****************NOTE: This is cv_set: ", i + 1, "*****************")
		
		model = IdenDKT(num_lstm_blocks = num_lstm_blocks, input_dim = input_dim, lstm_dim_ls = lstm_dim_ls,\
			num_layers_ls = num_layers_ls,dropout_ls = dropout_ls)
		
		train_path = cv_path_ls[i][0]
		val_path = cv_path_ls[i][1]
		test_path = cv_path_ls[i][2]
		print(train_path)
		print(val_path)
		print(test_path)

		# epoch, test_loss, test_auc, test_acc, test_rmse, model, test_outs = train_proc(train_path = train_path, val_path = val_path, test_path=test_path,\
		#                                             ckpt_path = ckpt_path_ls[i], n_question = int(hp_best['input_dim']/2), model = model, epochs=200, batch_size = 256,\
		#                                             shuffle = True, num_workers = 0, hp = hp_best, parallel=0, device_group = 'cuda:3')

		pred_percent_array, label_percent_array = utils.visual_proc(test_path=test_path, model_name = 'static-dkt',\
							ckpt_path = ckpt_path_ls[i], n_question = int(hp_best['input_dim']/2), model = model, epochs=1, batch_size = 256,\
							shuffle = True, num_workers = 0, hp = hp_best, parallel=0, device_group = 'cuda:1', test_skill_id=108)
		np.savez('./dkt_bertkt_share/static_dkt_pecent_correct_as09.npz', pred=pred_percent_array, label=label_percent_array)

	#     loss_ls.append(test_loss)
	#     auc_ls.append(test_auc)
	#     acc_ls.append(test_acc)
	#     rmse_ls.append(test_rmse)
	#     raw_pred_ls.append(test_outs)

	# test_loss_avg = sum(loss_ls)/len(loss_ls)
	# test_auc_avg = sum(auc_ls)/len(auc_ls)
	# test_acc_avg = sum(acc_ls)/len(acc_ls)
	# test_rmse_avg = sum(rmse_ls)/len(rmse_ls)

	# results_dict = hp_best.copy()
	# for key1 in results_dict:
	#     # print('key1')
	#     # print(key1)
	#     results_dict[key1] = str(results_dict[key1])

	# results_dict['epoch'] = str(epoch)
	# results_dict['test_loss_avg'] = str(test_loss_avg)
	# results_dict['test_auc_avg'] = str(test_auc_avg)
	# results_dict['test_acc_avg'] = str(test_acc_avg)
	# results_dict['test_rmse_avg'] = str(test_rmse_avg)

	# results_path = './nseq_DKT/cv_5_iden/assist_09_ckpt/assist_09.json'
	# with open(results_path, "w") as outfile:  
	#         json.dump(results_dict, outfile)
	# print('========================================= results dictionary saved to file! =========================================')

	# pred_path = './nseq_DKT/cv_5_iden/assist_09_ckpt/assist_09.csv'

	# start_time = time.time()
	# with open(pred_path, 'w') as f:
	#     for i, item in enumerate(raw_pred_ls):
	#         f.write('cv{}'.format(i+1))
	#         f.write('\n')
	#         f.write(','.join(map(str, item)))
	#         f.write('\n')
	# print('--- %s seconds ---' % (time.time()-start_time))
	
main()