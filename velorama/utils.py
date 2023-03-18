import numpy as np
import os
import torch
from scipy.stats import f
from scipy.sparse import csr_matrix
import statistics
import scanpy as sc
import scanpy.external as sce
from anndata import AnnData
import cellrank as cr
import scvelo as scv
import schema
from torch.nn.functional import normalize

from cellrank.tl.kernels import VelocityKernel

import torch
import torch.nn as nn

def construct_dag(adata,dynamics='rna_velocity',velo_mode='stochastic',proba=True,
				  n_neighbors=30,n_comps=50,use_time=False):

	"""Constructs the adjacency matrix for a DAG.
	Parameters
	----------
	adata: 'AnnData'
		AnnData object with rows corresponding to cells and columns corresponding 
		to genes.
	dynamics: {'rna_velocity','pseudotime','pseudotime_precomputed'} 
			  (default: rna_velocity)
		Dynamics used to orient and/or weight edges in the DAG of cells.
		If 'pseudotime_precomputed', the precomputed pseudotime values must be
		included as an observation category named 'pseudotime' in the included 
		AnnData object (e.g., adata.obs['pseudotime'] = [list of float]).
	velo_mode: {'stochastic','deterministic','dynamical'} (default: 'stochastic') 
		RNA velocity estimation using either the steady-state/deterministic, 
		stochastic, or dynamical model of transcriptional dynamics from scVelo
		(Bergen et al., 2020).
	proba: 'bool' (default: True)
		Whether to use the transition probabilities from CellRank (Lange et al., 2022) 
		in weighting the edges of the DAG or to discretize these probabilities by
		retaining only the top half of edges per cell.
	n_neighbors: 'int' (default: 15)
		Number of nearest neighbors to use in constructing a k-nearest
		neighbor graph for constructing a DAG if a custom DAG is not provided.
	n_comps: 'int' (default: 50)
		Number of principle components to compute and use for representing
		the gene expression profiles of cells.
	use_time: 'bool' (default: False)
		Whether to integrate time stamps in constructing the DAG. If True, time 
		stamps must be included as an observation category named 'time' in the 
		included AnnData object (e.g., adata.obs['time'] = [list of float]).
	"""

	sc.tl.pca(adata, n_comps=n_comps, svd_solver='arpack')
	if use_time:
		sqp = schema.SchemaQP(min_desired_corr=0.9,mode='affine',
							  params= {'decomposition_model': 'pca', 
							  'num_top_components': adata.obsm['X_pca'].shape[1]})
		adata.obsm['X_rep'] = sqp.fit_transform(adata.obsm['X_pca'],[adata.obs['time']],
						   ['numeric'],[1])
	else:
		adata.obsm['X_rep'] = adata.obsm['X_pca']

	if dynamics == 'pseudotime':
		A = construct_dag_pseudotime(adata.obsm['X_rep'],adata.uns['iroot'],
									 n_neighbors=n_neighbors).T

	elif dynamics == 'pseudotime_precomputed':
		A = construct_dag_pseudotime(adata.obsm['X_rep'],adata.uns['iroot'],
									 n_neighbors=n_neighbors,
									 pseudotime_algo='precomputed',
									 precomputed_pseudotime=adata.obs['pseudotime'].values).T

	elif dynamics == 'rna_velocity':
		scv.pp.moments(adata, n_neighbors=n_neighbors, use_rep='X_rep')
		if velo_mode == 'dynamical':
			scv.tl.recover_dynamics(adata)
		scv.tl.velocity(adata,mode=velo_mode)
		scv.tl.velocity_graph(adata)
		vk = VelocityKernel(adata).compute_transition_matrix()
		A = vk.transition_matrix
		A = A.toarray()

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

		for i in range(len(A)):
			for j in range(len(A)):
				if A[i][j] > 0 and A[j][i] > 0 and A[i][j] > A[j][i]:
					A[j][i] = 0

	A = construct_S(torch.FloatTensor(A))

	return A

def construct_dag_pseudotime(joint_feature_embeddings,iroot,n_neighbors=15,pseudotime_algo='dpt',
							 precomputed_pseudotime=None):
	
	"""Constructs the adjacency matrix for a DAG using pseudotime.
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
	precomputed_pseudotime: 'numpy.ndarray' or List (default: None)
		Precomputed pseudotime values for all cells.
	"""

	pseudotime,knn_graph = infer_knngraph_pseudotime(joint_feature_embeddings,iroot,
		n_neighbors=n_neighbors,pseudotime_algo=pseudotime_algo,
		precomputed_pseudotime=precomputed_pseudotime)
	dag_adjacency_matrix = dag_orient_edges(knn_graph,pseudotime)

	return dag_adjacency_matrix

def infer_knngraph_pseudotime(joint_feature_embeddings,iroot,n_neighbors=15,pseudotime_algo='dpt',
							  precomputed_pseudotime=None):

	adata = AnnData(joint_feature_embeddings)
	adata.obsm['X_joint'] = joint_feature_embeddings
	adata.uns['iroot'] = iroot

	if pseudotime_algo == 'dpt':
		sc.pp.neighbors(adata,use_rep='X_joint',n_neighbors=n_neighbors)
		sc.tl.dpt(adata)
		adata.obs['pseudotime'] = adata.obs['dpt_pseudotime'].values
		knn_graph = adata.obsp['distances'].astype(bool).astype(float)
	elif pseudotime_algo == 'precomputed':
		sc.pp.neighbors(adata,use_rep='X_joint',n_neighbors=n_neighbors)
		adata.obs['pseudotime'] = precomputed_pseudotime
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

def calculate_diffusion_lags(A,X,lag):

	if A == "linear":
		A = seq2dag(X.shape[1])

	ax = []
	cur = A
	for _ in range(lag):
		ax.append(torch.matmul(cur, X))
		cur = torch.matmul(A, cur)
		for i in range(len(cur)):
			cur[i][i] = 0

	return torch.stack(ax)

def load_gc_interactions(name,results_dir,lam_list,hidden_dim=16,lag=5,penalty='H',
						 dynamics='rna_velocity',seed=0,ignore_lag=False):
	
	config_name = '{}.seed{}.h{}.{}.lag{}.{}'.format(name,seed,hidden_dim,penalty,lag,dynamics)

	all_lags = []
	for lam in lam_list:
		if ignore_lag:
			file_name = '{}.seed{}.lam{}.h{}.{}.lag{}.{}.ignore_lag.pt'.format(name,seed,lam,hidden_dim,penalty,lag,dynamics)
			file_path = os.path.join(results_dir,config_name,file_name)
			gc_lag = torch.load(file_path)
			gc_lag = gc_lag.unsqueeze(-1)
		else:
			file_name = '{}.seed{}.lam{}.h{}.{}.lag{}.{}.pt'.format(name,seed,lam,hidden_dim,penalty,lag,dynamics)
			file_path = os.path.join(results_dir,config_name,file_name)
			gc_lag = torch.load(file_path)
		all_lags.append(gc_lag.detach())

	all_lags = torch.stack(all_lags)

	return all_lags

def lor(x, y):
	return x + y

def estimate_interactions(all_lags,lag=5,lower_thresh=0.01,upper_thresh=0.95,
						  binarize=False,l2_norm=False):

	all_interactions = []
	for i in range(len(all_lags)):
		for j in range(lag):

			nnz_percent = (all_lags[i,:,:,j] != 0).float().mean().data.numpy()

			if nnz_percent > lower_thresh and nnz_percent < upper_thresh:
				interactions = all_lags[i,:,:,j]

				if l2_norm:
					interactions = normalize(interactions,dim=(0,1))
				if binarize:
					interactions = (interactions != 0).float()

				all_interactions.append(interactions)
	return torch.stack(all_interactions).mean(0)

def estimate_lags(all_lags,lag,lower_thresh=0.01,upper_thresh=1.):
	
	retained_interactions = []
	for i in range(len(all_lags)):
		nnz_percent = (all_lags[i] != 0).float().mean().data.numpy()
		if nnz_percent > lower_thresh and nnz_percent < upper_thresh:            
			retained_interactions.append(all_lags[i])
	retained_interactions = torch.stack(retained_interactions)

	est_lags = normalize(retained_interactions,p=1,dim=-1).mean(0)
	return (est_lags*(torch.arange(lag)+1)).sum(-1)