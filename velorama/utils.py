import numpy as np
import os
import torch
from scipy.stats import f
from scipy.sparse import csr_matrix
import scanpy as sc
import scanpy.external as sce
from anndata import AnnData
import cellrank as cr
from cellrank.tl.kernels import VelocityKernel

import torch
import torch.nn as nn


def construct_dag(adata,dynamics='rna_velocity',proba=True):
	if dynamics == 'pseudotime':
		sc.tl.pca(adata, svd_solver='arpack')
		A = construct_dag_pseudotime(adata.obsm['X_pca'], adata.uns['iroot'])
		A = A.T
		A = construct_S(torch.FloatTensor(A))

	elif dynamics == 'rna_velocity':
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
		if not proba:
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

	return A

def construct_dag_pseudotime(joint_feature_embeddings,iroot,n_neighbors=15,pseudotime_algo='dpt'):
	
	"""Constructs the adjacency matrix for a DAG.
	Parameters
	----------
	joint_feature_embeddings: 'numpy.ndarray' (default: None)
		Matrix of low dimensional embeddings with rows corresponding
		to observations and columns corresponding to feature embeddings
		for constructing a DAG if a custom DAG is not provided.
	iroot: 'int' (default: None)
		Index of root cell for inferring pseudotime for constructing a DAG 
		if a custom DAG is not provided.
	n_neighbors: 'int' (default: 15)
		Number of nearest neighbors to use in constructing a k-nearest
		neighbor graph for constructing a DAG if a custom DAG is not provided.
	pseudotime_algo: {'dpt','palantir'} 
		Pseudotime algorithm to use for constructing a DAG if a custom DAG 
		is not provided. 'dpt' and 'palantir' perform the diffusion pseudotime
		(Haghverdi et al., 2016) and Palantir (Setty et al., 2019) algorithms, 
		respectively.
	"""

	pseudotime,knn_graph = infer_knngraph_pseudotime(joint_feature_embeddings,iroot,
		n_neighbors=n_neighbors,pseudotime_algo=pseudotime_algo)
	dag_adjacency_matrix = dag_orient_edges(knn_graph,pseudotime)

	return dag_adjacency_matrix

def infer_knngraph_pseudotime(joint_feature_embeddings,iroot,n_neighbors=15,pseudotime_algo='dpt'):

	adata = AnnData(joint_feature_embeddings)
	adata.obsm['X_joint'] = joint_feature_embeddings
	adata.uns['iroot'] = iroot

	if pseudotime_algo == 'dpt':
		sc.pp.neighbors(adata,use_rep='X_joint',n_neighbors=n_neighbors)
		sc.tl.dpt(adata)
		adata.obs['pseudotime'] = adata.obs['dpt_pseudotime'].values
		knn_graph = adata.obsp['distances'].astype(bool).astype(float)
	elif pseudotime_algo == 'palantir':
		sc.pp.neighbors(adata,use_rep='X_joint',n_neighbors=n_neighbors)
		sce.tl.palantir(adata, knn=n_neighbors,use_adjacency_matrix=True,
			distances_key='distances')
		pr_res = sce.tl.palantir_results(adata,
			early_cell=adata.obs.index.values[adata.uns['iroot']],
			ms_data='X_palantir_multiscale')
		adata.obs['pseudotime'] = pr_res.pseudotime
		knn_graph = adata.obsp['distances'].astype(bool).astype(float)

	return adata.obs['pseudotime'].values,knn_graph

def dag_orient_edges(adjacency_matrix,pseudotime):

	A = adjacency_matrix.astype(bool).astype(float)
	D = -1*np.sign(pseudotime[:,None] - pseudotime).T
	D = (D == 1).astype(float)
	D = (A.toarray()*D).astype(bool).astype(float)

	return D

def construct_S(D):
		
	S = D.clone()
	D_sum = D.sum(0)
	D_sum[D_sum == 0] = 1
	
	S = (S/D_sum)
	S = S.T
	
	return S

def normalize_adjacency(D):
		
	S = D.clone()
	D_sum = D.sum(0)
	D_sum[D_sum == 0] = 1
	
	S = (S/D_sum)
	
	return S

def seq2dag(N):
	A = torch.zeros(N, N)
	for i in range(N - 1):
		A[i][i + 1] = 1
	return A

def activation_helper(activation, dim=None):
	if activation == 'sigmoid':
		act = nn.Sigmoid()
	elif activation == 'tanh':
		act = nn.Tanh()
	elif activation == 'relu':
		act = nn.ReLU()
	elif activation == 'leakyrelu':
		act = nn.LeakyReLU()
	elif activation is None:
		def act(x):
			return x
	else:
		raise ValueError('unsupported activation: %s' % activation)
	return act

def calculate_AX(A,X,lag):

	if A == "linear":
		A = seq2dag(X.shape[1])
	S = construct_S(A)

	ax = []
	cur = S
	for _ in range(lag):
		ax.append(torch.matmul(cur, X))
		cur = torch.matmul(S, cur)
		for i in range(len(cur)):
			cur[i][i] = 0

	return torch.stack(ax)

def load_gc_interactions(name,root_dir,lam_list,hidden_dim=16,lag=5,penalty='H',
						 dynamics='rna_velocity',seed=0,ignore_lag=False):
	
	config_name = '{}.seed{}.h{}.{}.lag{}.{}'.format(name,seed,hidden_dim,penalty,lag,dynamics)

	all_lags = []
	for lam in lam_list:
		if ignore_lag:
			file_name = '{}.seed{}.lam{}.h{}.{}.lag{}.{}.ignore_lag.pt'.format(name,seed,lam,hidden_dim,penalty,lag,dynamics)
			file_path = os.path.join(root_dir,'results',config_name,file_name)
			gc_lag = torch.load(file_path)
			gc_lag = gc_lag.unsqueeze(-1)
		else:
			file_name = '{}.seed{}.lam{}.h{}.{}.lag{}.{}.pt'.format(name,seed,lam,hidden_dim,penalty,lag,dynamics)
			file_path = os.path.join(root_dir,'results',config_name,file_name)
			gc_lag = torch.load(file_path)
		all_lags.append(gc_lag.detach())

	all_lags = torch.stack(all_lags)

	return all_lags

def lor(x, y):
	return x + y

def estimate_interactions(all_lags,lag=5,lower_thresh=0.01,upper_thresh=0.95):
	
	gc = torch.swapaxes(all_lags,-1,1)

	rgc = []
	for i in range(lag):
		rmats = []
		for j in range(len(gc)):
			pnzero = (gc[j][i] > 1e-8).float().mean()
			if pnzero >= lower_thresh and pnzero <= upper_thresh:
				rmats.append(gc[j][i])
		rgc.append(rmats)

	fgc = torch.zeros(gc.shape[2:])
	for l in range(lag):
		if len(rgc[l]):
			fgc = lor(fgc, rgc[l][0])
	
	return fgc.T

def estimate_lags(all_lags,lag,thresh=1e-8,lower_thresh=0.01,upper_thresh=0.95):
	
	softmax = nn.Softmax(dim=-1)

	# consider only lambda values resulting GC matrices with between 1% and 95% non-zero values
	inds2keep = [i for i,gc in enumerate(all_lags) if 
				 (torch.norm(gc,dim=-1) > thresh).float().mean() <= upper_thresh and
				 (torch.norm(gc,dim=-1) > thresh).float().mean() >= lower_thresh]
	mats = (abs(all_lags[inds2keep]) > thresh).float() 
		
	for i in range(len(mats) - 2, -1, -1):
		mats[i] = lor(mats[i], mats[i + 1])
	bin_lags = abs(torch.FloatTensor(mats[0]))

	bin_lags = softmax(bin_lags)
	return (bin_lags[:,:,[lag-1]].mean(-1)-bin_lags[:,:,[0]].mean(-1)).squeeze()