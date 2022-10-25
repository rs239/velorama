#!/usr/bin/env python

### Authors: Anish Mudide (amudide), Alex Wu (alexw16), Rohit Singh (rs239)
### 2022
### MIT Licence
###

import os
import numpy as np
import scanpy as sc
import argparse
import time
import ray
from ray import tune
import statistics
import scvelo as scv
import pandas as pd

from .models import *
from .train import *
from .utils import *


def execute_cmdline():

	parser = argparse.ArgumentParser()
	parser.add_argument('-n','--name',dest='name',type=str,default='velorama_run',help='substring to have in our output files')
	parser.add_argument('-ds','--dataset',dest='dataset',type=str)
	parser.add_argument('-dyn','--dyn',dest='dynamics',type=str,default='pseudotime', choices=['pseudotime','rna_velocity'])
	parser.add_argument('-dev','--device',dest='device',type=str,default='cpu')
	parser.add_argument('-s','--seed',dest='seed',type=int,default=0,help='Random seed. Set to 0,1,2 etc.')
	parser.add_argument('-lmr','--lam_ridge',dest='lam_ridge',type=float,default=0., help='Currenty unsupported')
	parser.add_argument('-p','--penalty',dest='penalty',type=str,default="H")
	parser.add_argument('-l','--lag',dest='lag',type=int,default=5)
	parser.add_argument('-hd', '--hidden',dest='hidden',type=int,default=32)
	parser.add_argument('-mi','--max_iter',dest='max_iter',type=int,default=1000)
	parser.add_argument('-lr','--learning_rate',dest='learning_rate',type=float,default=0.0001)
	parser.add_argument('-pr','--proba',dest='proba',type=int,default=1)
	parser.add_argument('-ce','--check_every',dest='check_every',type=int,default=10)
	parser.add_argument('-rd','--root_dir',dest='root_dir',type=str)
	parser.add_argument('-ls','--lam_start',dest='lam_start',type=float,default=-2)
	parser.add_argument('-le','--lam_end',dest='lam_end',type=float,default=1)
	parser.add_argument('-xn','--x_norm',dest='x_norm',type=str,default='zscore',choices=['none','zscore','to_count:zscore'])
	parser.add_argument('-nl','--num_lambdas',dest='num_lambdas',type=int,default=19)

	args = parser.parse_args()

	results_dir = os.path.join(args.root_dir,'results')
	if not os.path.exists(results_dir):
		os.mkdir(results_dir)

	adata = sc.read(os.path.join(args.root_dir,'{}.h5ad'.format(args.dataset)))
	target_genes = adata.var.index.values[adata.var['is_target']]
	reg_genes = adata.var.index.values[adata.var['is_reg']]

	if args.x_norm == 'zscore':

		print('Normalizing data: 0 mean, 1 SD')
		X_orig = adata[:,adata.var['is_reg']].X.toarray().copy()
		std = X_orig.std(0)
		std[std == 0] = 1
		X = torch.FloatTensor(X_orig-X_orig.mean(0))/std

		Y_orig = adata[:,adata.var['is_target']].X.toarray().copy()
		std = Y_orig.std(0)
		std[std == 0] = 1
		Y = torch.FloatTensor(Y_orig-Y_orig.mean(0))/std	

	elif args.x_norm == 'to_count:zscore':

		print('Use counts: 0 mean, 1 SD')
		X_orig = adata[:,adata.var['is_reg']].X.toarray().copy()
		X_orig = 2**X_orig-1
		std = X_orig.std(0)
		std[std == 0] = 1
		X = torch.FloatTensor(X_orig-X_orig.mean(0))/std

		Y_orig = adata[:,adata.var['is_target']].X.toarray().copy()
		Y_orig = 2**Y_orig-1
		std = Y_orig.std(0)
		std[std == 0] = 1
		Y = torch.FloatTensor(Y_orig-Y_orig.mean(0))/std

	else:

		assert args.x_norm == 'none'				
		X = torch.FloatTensor(adata[:,adata.var['is_reg']].X.toarray())
		Y = torch.FloatTensor(adata[:,adata.var['is_target']].X.toarray())

	print('# of Regs: {}, # of Targets: {}'.format(X.shape[1],Y.shape[1]))

	print('Constructing DAG...')
	A = construct_dag(adata,dynamics=args.dynamics,proba=args.proba)
	A = torch.FloatTensor(A)
	AX = calculate_AX(A,X,args.lag)

	dir_name = '{}.seed{}.h{}.{}.lag{}.{}'.format(args.name,args.seed,args.hidden,args.penalty,args.lag,args.dynamics)

	if not os.path.exists(os.path.join(results_dir,dir_name)):
		os.mkdir(os.path.join(results_dir,dir_name))

	ray.init(object_store_memory=10**9)

	total_start = time.time()
	lam_list = np.round(np.logspace(args.lam_start, args.lam_end, num=args.num_lambdas),4).tolist()

	config = {'name': args.name,
			  'AX': AX,
			  'Y': Y,
			  'seed': args.seed,
			  'lr': args.learning_rate,
			  'lam': tune.grid_search(lam_list),
			  'lam_ridge': args.lam_ridge,
			  'penalty': args.penalty,
			  'lag': args.lag,
			  'hidden': [args.hidden],
			  'max_iter': args.max_iter,
			  'device': args.device,
			  'lookback': 5,
			  'check_every': args.check_every,
			  'verbose': True,
			  'dynamics': args.dynamics,
			  'results_dir': results_dir,
			  'dir_name': dir_name}

	resources_per_trial = {"cpu": 1, "gpu": 0.2, "memory": 2 * 1024 * 1024 * 1024}
	analysis = tune.run(train_model,resources_per_trial=resources_per_trial,config=config,
						local_dir=os.path.join(args.root_dir,'results'))
	
	# aggregate results
	all_lags = load_gc_interactions(args.name,args.root_dir,lam_list,hidden_dim=args.hidden,
									lag=args.lag,penalty=args.penalty,
						 			dynamics=args.dynamics,seed=args.seed,ignore_lag=False)

	gc_mat = estimate_interactions(all_lags,lag=args.lag)
	gc_df = pd.DataFrame(gc_mat.cpu().data.numpy(),index=target_genes,columns=reg_genes)
	gc_df.to_csv(os.path.join(results_dir,'velorama.interactions.tsv'),sep='\t')

	lag_mat = estimate_lags(all_lags,lag=args.lag)
	lag_df = pd.DataFrame(lag_mat.cpu().data.numpy(),index=target_genes,columns=reg_genes)
	lag_df.to_csv(os.path.join(results_dir,'velorama.lags.tsv'),sep='\t')

	print('Total time:',time.time()-total_start)
	np.savetxt(os.path.join(results_dir,dir_name + '.time.txt'),np.array([time.time()-total_start]))

if __name__ == "__main__":
	execute_cmdline()
