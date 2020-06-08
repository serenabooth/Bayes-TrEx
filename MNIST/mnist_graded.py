
from __future__ import division, print_function

import torch, pyro, argparse, os
import pyro.distributions as dist
from pyro.infer.mcmc import MCMC, HMC, NUTS
import numpy as np
from multiprocessing import set_start_method, Pool

from nn_models import *

def program_arbitrary(nn_model, p_tgt, std=0.05):
	'''
	a probabilistic model for enforcing p = p_tgt
	sample u_i ~ No(p_i, std)
	then the posterior is p( z | u_i=p_tgt[i] )
	'''
	if nn_model.device == 'cuda':
		typ = torch.cuda.FloatTensor
	elif nn_model.device == 'cpu':
		typ = torch.FloatTensor
	torch.set_default_tensor_type(typ)

	std = torch.tensor(std).float()
	latent_dim = nn_model.latent_dim
	loc = torch.zeros(latent_dim)
	cov = torch.eye(latent_dim)
	z = pyro.sample('z', dist.MultivariateNormal(loc, cov))
	prob = nn_model.predict_from_latent(z)
	N = len(prob)
	us = []
	for i in range(N):
		us.append(pyro.sample('u_%i'%i, dist.Normal(prob[i], std), obs=torch.tensor(p_tgt[i])))

def setting(args):
	folder = 'results/graded/%s_%s_%s'%(args.nn_model, args.dataset, args.type)
	os.makedirs(folder, exist_ok=True)
	if args.nn_model == 'vae':
		nn_model = VAEModel(args.dataset)
	elif args.nn_model == 'gan':
		nn_model = GANModel(args.dataset)
	a, b = map(int, args.type.split('v'))
	return nn_model, a, b, folder

def run(param):
	nn_model, p_tgt, save_fn, args = param
	if (not args.overwrite) and os.path.isfile(save_fn):
		print(save_fn+' already exists!')
		return
	if not os.path.isfile(save_fn):
		fo = open(save_fn, 'w') # write the file first to signal working on it.
		fo.write('\n')
		fo.close()
	nuts = NUTS(program_arbitrary)
	mcmc = MCMC(nuts, num_samples=args.num_samples, 
		warmup_steps=args.num_warmups, num_chains=args.num_chains)
	mcmc.run(nn_model, p_tgt)
	zs = mcmc.get_samples()['z'].detach().cpu().numpy()
	np.savetxt(save_fn, zs)


if __name__ == '__main__':
	set_start_method('spawn')

	# fashion_mnist labels
	# [0: 'T-shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 
	#  5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot']

	parser = argparse.ArgumentParser()
	parser.add_argument('--num-samples', type=int, default=1000)
	parser.add_argument('--num-warmups', type=int, default=1000)
	parser.add_argument('--num-chains', type=int, default=1)
	parser.add_argument('--overwrite', action='store_true')
	parser.add_argument('--nn-model', default='vae')
	parser.add_argument('--dataset')
	parser.add_argument('--type')
	args = parser.parse_args()
	nn_model, a, b, folder = setting(args)
	p_tgts = []
	save_fns = []
	for i in np.linspace(0, 1, 11):
		p = [0] * 10
		p[a] = i
		p[b] = 1-i
		p_tgts.append(p)
		save_fn = '%s/%0.1f_%i_%0.1f_%i.txt'%(folder, i, a, 1-i, b)
		save_fns.append(save_fn)
	nn_models = [nn_model] * 11
	argss = [args] * 11
	params = zip(nn_models, p_tgts, save_fns, argss)
	pool = Pool(4)
	pool.map(run, params)
	
