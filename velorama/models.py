#!/usr/bin/env python

### Authors: Anish Mudide (amudide), Alex Wu (alexw16), Rohit Singh (rs239)
### 2022
### MIT Licence
###
### Credit: parts of this code make use of the code from Tank et al.'s "Neural Granger Causality"
###    - https://github.com/iancovert/Neural-GC
###    - https://arxiv.org/abs/1802.05842

import torch
import torch.nn as nn

from utils import activation_helper

class VeloramaMLP(nn.Module):

	def __init__(self, n_targets, n_regs, lag, hidden, device, activation):
		super(VeloramaMLP, self).__init__()
		self.activation = activation_helper(activation)
		self.hidden = hidden
		self.lag = lag
		self.device = device

		# set up first layer
		layer = nn.Conv1d(n_regs, hidden[0]*n_targets, lag)
		modules = [layer]
		
		# set up subsequent layers
		for d_in, d_out in zip(hidden, hidden[1:] + [1]):
			layer = nn.Conv1d(d_in*n_targets, d_out*n_targets, 1, groups=n_targets)
			modules.append(layer)

		# Register parameters.
		self.layers = nn.ModuleList(modules)

	def forward(self,AX):
		
		# first layer
		ret = 0
		for i in range(self.lag):
			ret = ret + torch.matmul(AX[i], self.layers[0].weight[:, :, self.lag - 1 - i].T)
		ret = ret + self.layers[0].bias
		
		# subsequent layers
		ret = ret.T
		for i, fc in enumerate(self.layers):
			if i == 0:
				continue
			ret = self.activation(ret)
			ret = fc(ret)
		
		return ret.T

	def GC(self, threshold=True, ignore_lag=True):
		'''
		Extract learned Granger causality.

		Args:
		  threshold: return norm of weights, or whether norm is nonzero.
		  ignore_lag: if true, calculate norm of weights jointly for all lags.

		Returns:
		  GC: (p x p) or (p x p x lag) matrix. In first case, entry (i, j)
			indicates whether variable j is Granger causal of variable i. In
			second case, entry (i, j, k) indicates whether it's Granger causal
			at lag k.
		'''

		W = self.layers[0].weight
		W = W.reshape(-1,self.hidden[0],W.shape[1],W.shape[2])

		if ignore_lag:
			GC = torch.norm(W, dim=(1, 3))
		else:
			GC = torch.norm(W, dim=1)

		if threshold:
			return (GC > 0).int()
		else:
			return GC

class VeloramaMLPTarget(nn.Module):

	def __init__(self, n_targets, n_regs, lag, hidden, device, activation):
		super(VeloramaMLPTarget, self).__init__()
		self.activation = activation_helper(activation)
		self.hidden = hidden
		self.lag = lag
		self.device = device

		# set up first layer
		layer = nn.Conv1d(n_regs, hidden[0]*n_targets, lag)
		modules = [layer]
		
		# set up first layers (target variables)
		target_modules = [nn.Conv1d(n_targets, hidden[0]*n_targets,1,groups=n_targets,bias=False)
							  for _ in range(lag)]
		self.target_layers = nn.ModuleList(target_modules)

		# set up subsequent layers
		for d_in, d_out in zip(hidden, hidden[1:] + [1]):
			layer = nn.Conv1d(d_in*n_targets, d_out*n_targets, 1, groups=n_targets)
			modules.append(layer)

		# Register parameters.
		self.layers = nn.ModuleList(modules)

	def forward(self,AX,AY):

		# first layer
		ret = 0
		for i in range(self.lag):
			ret = ret + torch.matmul(AX[i], self.layers[0].weight[:, :, self.lag - 1 - i].T)
		ret = ret + self.layers[0].bias
		
		# include contributions of target variables
		ret = ret.T
		for i in range(self.lag):
			ret = ret + self.target_layers[self.lag - 1 - i](AY[i].T)
			
		# subsequent layers
		for i, fc in enumerate(self.layers):
			if i == 0:
				continue
			ret = self.activation(ret)
			ret = fc(ret)

		return ret.T

	def GC(self, threshold=True, ignore_lag=True):
		'''
		Extract learned Granger causality.

		Args:
		  threshold: return norm of weights, or whether norm is nonzero.
		  ignore_lag: if true, calculate norm of weights jointly for all lags.

		Returns:
		  GC: (p x p) or (p x p x lag) matrix. In first case, entry (i, j)
			indicates whether variable j is Granger causal of variable i. In
			second case, entry (i, j, k) indicates whether it's Granger causal
			at lag k.
		'''

		W = self.layers[0].weight
		W = W.reshape(-1,self.hidden[0],W.shape[1],W.shape[2])

		if ignore_lag:
			GC = torch.norm(W, dim=(1, 3))
		else:
			GC = torch.norm(W, dim=1)

		if threshold:
			return (GC > 0).int()
		else:
			return GC

def prox_update(network, lam, lr, penalty):
	'''
	Perform in place proximal update on first layer weight matrix.
	Args:
	  network: MLP network.
	  lam: regularization parameter.
	  lr: learning rate.
	  penalty: one of GL (group lasso), GSGL (group sparse group lasso),
		H (hierarchical).
	'''
	W = network.layers[0].weight
	hidden, p, lag = W.shape
	if penalty == 'GL':
		norm = torch.norm(W, dim=(0, 2), keepdim=True)
		W.data = ((W / torch.clamp(norm, min=(lr * lam)))
				  * torch.clamp(norm - (lr * lam), min=0.0))
	elif penalty == 'GSGL':
		norm = torch.norm(W, dim=0, keepdim=True)
		W.data = ((W / torch.clamp(norm, min=(lr * lam)))
				  * torch.clamp(norm - (lr * lam), min=0.0))
		norm = torch.norm(W, dim=(0, 2), keepdim=True)
		W.data = ((W / torch.clamp(norm, min=(lr * lam)))
				  * torch.clamp(norm - (lr * lam), min=0.0))
	elif penalty == 'H':
		# Lowest indices along third axis touch most lagged values.
		for i in range(lag):
			norm = torch.norm(W[:, :, :(i + 1)], dim=(0, 2), keepdim=True)
			W.data[:, :, :(i+1)] = (
				(W.data[:, :, :(i+1)] / torch.clamp(norm, min=(lr * lam)))
				* torch.clamp(norm - (lr * lam), min=0.0))
	else:
		raise ValueError('unsupported penalty: %s' % penalty)

def prox_update_target(network, lam, lr, penalty):
	'''
	Perform in place proximal update on first layer weight matrix.
	Args:
	  network: MLP network.
	  lam: regularization parameter.
	  lr: learning rate.
	  penalty: one of GL (group lasso), GSGL (group sparse group lasso),
		H (hierarchical).
	'''
	W = network.layers[0].weight
	hidden, p, lag = W.shape

	if penalty == 'GL':
		norm = torch.norm(W, dim=(0, 2), keepdim=True)
		W.data = ((W / torch.clamp(norm, min=(lr * lam)))
				  * torch.clamp(norm - (lr * lam), min=0.0))
	elif penalty == 'GSGL':
		norm = torch.norm(W, dim=0, keepdim=True)
		W.data = ((W / torch.clamp(norm, min=(lr * lam)))
				  * torch.clamp(norm - (lr * lam), min=0.0))
		norm = torch.norm(W, dim=(0, 2), keepdim=True)
		W.data = ((W / torch.clamp(norm, min=(lr * lam)))
				  * torch.clamp(norm - (lr * lam), min=0.0))
	elif penalty == 'H':
		# Lowest indices along third axis touch most lagged values.
		for i in range(lag):
			W = network.layers[0].weight
			target_W = torch.stack([network.target_layers[j].weight for j
									in range(len(network.target_layers))]).squeeze(-1)
			target_W = torch.swapaxes(torch.swapaxes(target_W,0,1),1,2)
			W_concat = torch.cat([W.data[:,:,:(i+1)],target_W.data[:,:,:(i+1)]],dim=1)
			norm = torch.norm(W_concat[:,:,:(i+1)], dim=(0, 2), keepdim=True)

			# update regulator weights
			W.data[:, :, :(i+1)] = (
				(W.data[:, :, :(i+1)] / torch.clamp(norm[:,0:-1], min=(lr * lam)))
				* torch.clamp(norm[:,0:-1] - (lr * lam), min=0.0))

			# update target weights
			for j in range(i+1):
				W_t = network.target_layers[j].weight
				W_t.data = ((W_t.data / torch.clamp(norm[:,-1:], min=(lr * lam)))
				* torch.clamp(norm[:,-1:] - (lr * lam), min=0.0))
	else:
		raise ValueError('unsupported penalty: %s' % penalty)

# def prox_update_new(network, lam, lr, penalty):
# 	'''
# 	Perform in place proximal update on first layer weight matrix.

# 	Args:
# 	  network: MLP network.
# 	  lam: regularization parameter.
# 	  lr: learning rate.
# 	  penalty: one of GL (group lasso), GSGL (group sparse group lasso),
# 		H (hierarchical).
# 	'''

# 	W = network.layers[0].weight
# 	hidden, p, lag = W.shape

# 	W_copy = torch.clone(W)
# 	W_copy = W.reshape(-1,network.hidden[0],W.shape[1],W.shape[2])
	
# 	if penalty == 'GL':

# 		norm = torch.norm(W_copy[:, :, :, :(i + 1)], dim=(1, 3), keepdim=True)
# 		W_copy.data = W_copy

# 		norm = torch.norm(W, dim=(0, 2), keepdim=True)
# 		W.data = ((W / torch.clamp(norm, min=(lr * lam)))
# 				  * torch.clamp(norm - (lr * lam), min=0.0))
# 	# elif penalty == 'GSGL':
# 	# 	norm = torch.norm(W, dim=0, keepdim=True)
# 	# 	W.data = ((W / torch.clamp(norm, min=(lr * lam)))
# 	# 			  * torch.clamp(norm - (lr * lam), min=0.0))
# 	# 	norm = torch.norm(W, dim=(0, 2), keepdim=True)
# 	# 	W.data = ((W / torch.clamp(norm, min=(lr * lam)))
# 	# 			  * torch.clamp(norm - (lr * lam), min=0.0))
# 	elif penalty == 'H':
# 		# Lowest indices along third axis touch most lagged values.
# 		for i in range(lag):
# 			norm = torch.norm(W_copy[:, :, :, :(i + 1)], dim=(1, 3), keepdim=True)
# 			W_copy.data[:, :, :, :(i+1)] = (
# 				(W_copy.data[:, :, :, :(i+1)] / torch.clamp(norm, min=(lr * lam)))
# 				* torch.clamp(norm - (lr * lam), min=0.0))
# 		W.data = W_copy.data.reshape(W.shape)

# 	else:
# 		raise ValueError('unsupported penalty: %s' % penalty)

def regularize(network, lam, penalty):
	'''
	Calculate regularization term for first layer weight matrix.
	Args:
	  network: MLP network.
	  penalty: one of GL (group lasso), GSGL (group sparse group lasso),
		H (hierarchical).
	'''
	W = network.layers[0].weight
	hidden, p, lag = W.shape
	if penalty == 'GL':
		return lam * torch.sum(torch.norm(W, dim=(0, 2)))
	elif penalty == 'GSGL':
		return lam * (torch.sum(torch.norm(W, dim=(0, 2)))
					  + torch.sum(torch.norm(W, dim=0)))
	elif penalty == 'H':
		# Lowest indices along third axis touch most lagged values.
		return lam * sum([torch.sum(torch.norm(W[:, :, :(i+1)], dim=(0, 2)))
						  for i in range(lag)])
	else:
		raise ValueError('unsupported penalty: %s' % penalty)
		
def regularize_target(network, lam, penalty):
	'''
	Calculate regularization term for first layer weight matrix.
	Args:
	  network: MLP network.
	  penalty: one of GL (group lasso), GSGL (group sparse group lasso),
		H (hierarchical).
	'''
	W = network.layers[0].weight

	hidden, p, lag = W.shape
	if penalty == 'GL':
		return lam * torch.sum(torch.norm(W, dim=(0, 2)))
	elif penalty == 'GSGL':
		return lam * (torch.sum(torch.norm(W, dim=(0, 2)))
					  + torch.sum(torch.norm(W, dim=0)))
	elif penalty == 'H':
		# Lowest indices along third axis touch most lagged values.
		# return lam * sum([torch.sum(torch.norm(W[:, :, :(i+1)], dim=(0, 2)))
		# 				  for i in range(lag)])
		target_W = torch.stack([network.target_layers[i].weight for i 
								in range(len(network.target_layers))]).squeeze(-1)
		target_W = torch.swapaxes(torch.swapaxes(target_W,0,1),1,2)

		return lam * sum([torch.sum(torch.norm(torch.cat([W.data[:,:,:(i+1)],target_W.data[:,:,:(i+1)]],dim=1), 
			dim=(0, 2))) for i in range(lag)])
	else:
		raise ValueError('unsupported penalty: %s' % penalty)

# def regularize_new(network, lam, penalty):
# 	'''
# 	Calculate regularization term for first layer weight matrix.

# 	Args:
# 	  network: MLP network.
# 	  penalty: one of GL (group lasso), GSGL (group sparse group lasso),
# 		H (hierarchical).
# 	'''
# 	W = network.layers[0].weight
# 	hidden, p, lag = W.shape
# 	W = W.reshape(-1,network.hidden[0],W.shape[1],W.shape[2])

# 	if penalty == 'GL':
# 		return lam * torch.sum(torch.norm(W, dim=(1, 3))).sum()
# 	# elif penalty == 'GSGL':
# 	# 	return lam * (torch.sum(torch.norm(W, dim=(0, 3)))
# 	# 				  + torch.sum(torch.norm(W, dim=0)))
# 	elif penalty == 'H':
# 		# Lowest indices along third axis touch most lagged values.
# 		return lam * sum([torch.sum(torch.norm(W[:, :, :, :(i+1)], dim=(1, 3)))
# 						  for i in range(lag)]).sum()
# 	else:
# 		raise ValueError('unsupported penalty: %s' % penalty)

def ridge_regularize(network, lam):
	'''Apply ridge penalty at all subsequent layers.'''
	return lam * sum([torch.sum(fc.weight ** 2) for fc in network.layers[1:]])


def restore_parameters(model, best_model):
	'''Move parameter values from best_model to model.'''
	for params, best_params in zip(model.parameters(), best_model.parameters()):
		params.data = best_params
