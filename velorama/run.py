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

from .models import *
from .train import *
from .utils import *

def execute_cmdline():

	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--method',dest='method',type=str,default='baseline')
	parser.add_argument('-ds','--dataset',dest='dataset',type=str)
	parser.add_argument('-dyn','--dyn',dest='dynamics',type=str,default='pseudotime')
	parser.add_argument('-dev','--device',dest='device',type=str,default='cpu')
	parser.add_argument('-s','--seed',dest='seed',type=int,default=0,help='random seed used for trial. set to 0,1,2 etc.')
	parser.add_argument('-lmr','--lam_ridge',dest='lam_ridge',type=float,default=0., help='Unsupported currently')
	parser.add_argument('-p','--penalty',dest='penalty',type=str,default="H")
	parser.add_argument('-l','--lag',dest='lag',type=int,default=5)
	parser.add_argument('-hd', '--hidden',dest='hidden',type=int,default=16)
	parser.add_argument('-mi','--max_iter',dest='max_iter',type=int,default=500)
	parser.add_argument('-lr','--learning_rate',dest='learning_rate',type=float,default=0.0001)
	parser.add_argument('-pr','--proba',dest='proba',type=int,default=1)
	parser.add_argument('-tol','--tolerance',dest='tolerance',type=float,default=0.01)
	parser.add_argument('-ce','--check_every',dest='check_every',type=int,default=100)
	parser.add_argument('-rd','--root_dir',dest='root_dir',type=str)
	parser.add_argument('-ls','--lam_start',dest='lam_start',type=float,default=-1)
	parser.add_argument('-le','--lam_end',dest='lam_end',type=float,default=1)
	parser.add_argument('-xn','--x_norm',dest='x_norm',type=str,default='none',choices=['none','zscore','to_count:zscore'])
	parser.add_argument('-nl','--num_lambdas',dest='num_lambdas',type=int,default=19)

	args = parser.parse_args()

	data_dir = os.path.join(args.root_dir,'datasets','preprocessed')
	gc_dir = os.path.join(args.root_dir,'results',args.dataset,'gc')

	if not os.path.exists(os.path.join(args.root_dir,'results',args.dataset)):
		os.mkdir(os.path.join(args.root_dir,'results',args.dataset))
	if not os.path.exists(gc_dir):
		os.mkdir(gc_dir)

	adata = sc.read(os.path.join(data_dir,'{}.h5ad'.format(args.dataset)))

	if args.dynamics == 'pseudotime':
		print('Inferring pseudotime transition matrix...')
		sc.tl.pca(adata, svd_solver='arpack')
		A = construct_dag(adata.obsm['X_pca'], adata.uns['iroot'])
		A = A.T
		A = construct_S(torch.FloatTensor(A))

	elif args.dynamics == 'rna_velocity':
		print('Inferring RNA velocity transition matrix...')

		scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
		scv.tl.velocity(adata)
		scv.tl.velocity_graph(adata)
		vk = VelocityKernel(adata).compute_transition_matrix()
		A = vk.transition_matrix
		A = A.toarray()
		for i in range(len(A)):
			for j in range(len(A)):
				if A[i][j] > 0 and A[j][i] > 0 and A[i][j] > A[j][i]:
					A[j][i] = 0

		# if proba is False (0), it won't use the probabilistic 
		# transition matrix
		if not args.proba:
			for i in range(len(A)):
				nzeros = []
				for j in range(len(A)):
					if A[i][j] > 0:
						nzeros.append(A[i][j])
				m = statistics.median(nzeros)
				for j in range(len(A)):
					if A[i][j] < m:
						A[i][j] = 0
					else:
						A[i][j] = 1

		A = construct_S(torch.FloatTensor(A))

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

	print('Performing diffusion...')
	A = torch.FloatTensor(A)
	AX = calculate_AX(A,X,args.lag)

	dir_name = '{}.seed{}.h{}.{}.lag{}.{}'.format(args.method,args.seed,args.hidden,args.penalty,args.lag,args.dynamics)

	if not os.path.exists(os.path.join(gc_dir,dir_name)):
		os.mkdir(os.path.join(gc_dir,dir_name))

	ray.init(object_store_memory=10**9)

	total_start = time.time()
	lam_list = np.round(np.logspace(args.lam_start, args.lam_end, num=args.num_lambdas),4).tolist()
	# lam_list = sorted(list(set(lam_list)))

	config = {'method': args.method,
			  'AX': AX,
			  'Y': Y,
			  'trial': args.seed,
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
			  'tol': args.tolerance,
			  'dynamics': args.dynamics,
			  'gc_dir': gc_dir,
			  'dir_name': dir_name}

	resources_per_trial = {"cpu": 1, "gpu": 0.2, "memory": 2 * 1024 * 1024 * 1024}
	analysis = tune.run(train_model,resources_per_trial=resources_per_trial,config=config,
						local_dir=os.path.join(args.root_dir,'results'))
	
	print('Total time:',time.time()-total_start)
	np.savetxt(os.path.join(gc_dir,dir_name + '.time.txt'),np.array([time.time()-total_start]))

if __name__ == "__main__":
	execute_cmdline()
