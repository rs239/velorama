#!/usr/bin/env python

### Authors: Anish Mudide (amudide), Alex Wu (alexw16), Rohit Singh (rs239)
### 2022
### MIT Licence
###


import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import time

from models import *
from utils import *

def train_model(config, checkpoint_dir = None):

	AX = config["AX"]
	AY = config["AY"]

	Y = config["Y"]

	name = config["name"]

	seed = config["seed"]
	lr = config["lr"]
	lam = config["lam"]
	lam_ridge = config["lam_ridge"]
	penalty = config["penalty"]
	lag = config["lag"]
	hidden = config["hidden"]
	max_iter = config["max_iter"]
	device = config["device"]
	lookback = config["lookback"]
	check_every = config["check_every"]
	verbose = config["verbose"]
	dynamics = config['dynamics']
		
	results_dir = config['results_dir']
	dir_name = config['dir_name']
	reg_target = config['reg_target']

	np.random.seed(seed)
	torch.manual_seed(seed)

	file_name = '{}.seed{}.lam{}.h{}.{}.lag{}.{}'.format(name,seed,np.round(lam,4),
				hidden[0],penalty,lag,dynamics)
	gc_path1 = os.path.join(results_dir,dir_name,file_name + '.pt')
	gc_path2 = os.path.join(results_dir,dir_name,file_name + '.ignore_lag.pt')

	if not os.path.exists(gc_path1) and not os.path.exists(gc_path2):
		
		num_regs = AX.shape[-1]
		num_targets = Y.shape[1]

		AX = AX.to(device)		
		Y = Y.to(device)

		if reg_target:
			vmlp = VeloramaMLPTarget(num_targets, num_regs, lag=lag, hidden=hidden, 
									 device=device, activation='relu')
			AY = AY.to(device)
		else:
			vmlp = VeloramaMLP(num_targets, num_regs, lag=lag, hidden=hidden, 
							   device=device, activation='relu')

		vmlp.to(device)

		'''Train model with ISTA.'''
		lag = vmlp.lag
		loss_fn = nn.MSELoss(reduction='none')
		train_loss_list = []

		# For early stopping.
		best_it = None
		best_loss = np.inf
		best_model = None

		# Calculate smooth error.
		if reg_target:
			preds = vmlp(AX,AY)
		else:
			preds = vmlp(AX)

		loss = loss_fn(preds,Y).mean(0).sum()

		# print('LOSS:---',loss)

		ridge = ridge_regularize(vmlp, lam_ridge) 
		smooth = loss + ridge

		variable_usage_list = []
		loss_list = []

		# For early stopping.
		train_loss_list = []
		best_it = None
		best_loss = np.inf
		best_model = None

		for it in range(max_iter):

			start = time.time()

			# Take gradient step.
			smooth.backward()
			for param in vmlp.parameters():
				param.data = param - lr * param.grad

			# Take prox step.
			if lam > 0:
				# for net in vmlp.networks:
				if reg_target:
					prox_update_target(vmlp, lam, lr, penalty)
				else:
					prox_update(vmlp, lam, lr, penalty)

			vmlp.zero_grad()

			# Calculate loss for next iteration.
			if reg_target:
				preds = vmlp(AX,AY)
			else:
				preds = vmlp(AX)

			loss = loss_fn(preds,Y).mean(0).sum()

			ridge = ridge_regularize(vmlp, lam_ridge)
			smooth = loss + ridge

			# Check progress.
			if (it + 1) % check_every == 0:

				if reg_target:
					nonsmooth = regularize_target(vmlp, lam, penalty)
				else:
					nonsmooth = regularize(vmlp, lam, penalty)
				mean_loss = (smooth + nonsmooth).detach()/Y.shape[1]

				variable_usage = torch.mean(vmlp.GC(ignore_lag=False).float())
				variable_usage_list.append(variable_usage)
				loss_list.append(mean_loss)

				# Check for early stopping.
				if mean_loss < best_loss:
					best_loss = mean_loss
					best_it = it
					best_model = deepcopy(vmlp)

					if verbose:
						print('Lam={}: Iter {}, {} sec'.format(lam,it+1,np.round(time.time()-start,2)),
							  '-----','Loss: %.2f' % mean_loss,', Variable usage = %.2f%%' % (100 * variable_usage)) # ,
							  # '|||','%.3f' % loss_crit,'%.3f' % variable_usage_crit)

				elif (it - best_it) == lookback * check_every:
					if verbose:
						print('EARLY STOP: Lam={}, Iter {}'.format(lam,it + 1))
					break

		if verbose:
			print('Lam={}: Completed in {} iterations'.format(lam,it+1))

		# Restore best model.
		restore_parameters(vmlp, best_model)

		if not os.path.exists(results_dir):
			os.mkdir(results_dir)
		if not os.path.exists(os.path.join(results_dir,dir_name)):
			os.mkdir(os.path.join(results_dir,dir_name))

		file_name = '{}.seed{}.lam{}.h{}.{}.lag{}.{}'.format(name,seed,np.round(lam,4),
					hidden[0],penalty,lag,dynamics)
		GC_lag = vmlp.GC(threshold=False, ignore_lag=False).cpu()
		torch.save(GC_lag, os.path.join(results_dir,dir_name,file_name + '.pt'))

		GC_lag = vmlp.GC(threshold=False, ignore_lag=True).cpu()
		torch.save(GC_lag, os.path.join(results_dir,dir_name,file_name + '.ignore_lag.pt'))
