
from __future__ import division, print_function

import torch, pyro, argparse, os
import pyro.distributions as dist
from pyro.infer.mcmc import MCMC, HMC, NUTS
import numpy as np
from multiprocessing import set_start_method

from nn_models import *

def program_avb(nn_model, a, b, std_diff=0.05, std_margin=0.05):
	'''
	a probabilistic model for enforcing p_a = p_b > p_i, for i != a, b
	let diff = |p_a - p_b|, and margin = min(p_a, p_b) - max(p_i)
	sample u_1 ~ No(diff, std_diff), u_2 ~ No(margin, std_margin)
	then the posterior is p( z | u_1=0, u_2=0.5 )
	'''
	if nn_model.device == 'cuda':
		typ = torch.cuda.FloatTensor
	elif nn_model.device == 'cpu':
		typ = torch.FloatTensor
	torch.set_default_tensor_type(typ)

	std_diff = torch.tensor(std_diff).float()
	std_margin = torch.tensor(std_margin).float()
	latent_dim = nn_model.latent_dim
	loc = torch.zeros(latent_dim)
	cov = torch.eye(latent_dim)
	z = pyro.sample('z', dist.MultivariateNormal(loc, cov))
	prob = nn_model.predict_from_latent(z)
	N = len(prob)
	assert a < N and b < N
	p_a = prob[a]
	p_b = prob[b]
	p_other = torch.stack([prob[i] for i in range(N) if i!=a and i!=b])
	diff = torch.abs(p_a - p_b)
	p_ab = torch.stack((p_a, p_b))
	margin = torch.min(p_ab) - torch.max(p_other)
	u_1 = pyro.sample('u1', dist.Normal(diff, std_diff), obs=torch.tensor(0.0))
	u_2 = pyro.sample('u2', dist.Normal(margin, std_margin), obs=torch.tensor(0.5))


def program_targeted(nn_model, label, std=0.05):
	'''
	a probabilistic model for enforcing p_label = 1
	sample u ~ No(p_label, std)
	then the posterior is p( z | u=1.0 )
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
	assert label < N
	p_l = prob[label]
	u = pyro.sample('u', dist.Normal(p_l, std), obs=torch.tensor(1.0))

def program_unif(nn_model, std=0.05):
	'''
	a probabilistic model for enforcing p_i = p_j, for all i, j
	let diff = max(p_i) - min(p_i) and sample u ~ No(diff, std)
	then the posterior is p( z | u=0 )
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
	diff = torch.max(prob) - torch.min(prob)
	u = pyro.sample('u', dist.Normal(diff, std), obs=torch.tensor(0.0))


def setting(args):
	save_dir = 'results/ambivalent/'
    # check if the directory exists; if not, make it
	if not os.path.isdir(save_dir):
		os.mkdir(save_dir)

	save_fn = save_dir + '%s_%s_%s.txt'%(
			   args.nn_model, args.dataset, args.type)
	if (not args.overwrite) and os.path.isfile(save_fn):
		print(save_fn+' already exists!')
		quit()
	if not os.path.isfile(save_fn):
		print ("hi: " + str(save_fn))
		fo = open(save_fn, "w+") # write the file first to signal working on it.
		fo.write('\n')
		fo.close()
	if 'v' in args.type:
		a, b = map(int, args.type.split('v'))
		program = program_avb
		run_args = (a, b)
	elif args.type=='unif':
		program = program_unif
		run_args = ()
	elif 'h' in args.type:
		program = program_targeted
		label = int(args.type[1:])
		run_args = (label,)
	if args.nn_model == 'vae':
		nn_model = VAEModel(args.dataset)
	elif args.nn_model == 'gan':
		nn_model = GANModel(args.dataset)
	return nn_model, program, run_args, save_fn

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
	parser.add_argument('--nn-model')
	parser.add_argument('--dataset')
	parser.add_argument('--type')
	args = parser.parse_args()
	nn_model, program, run_args, save_fn = setting(args)

	nuts = NUTS(program)
	mcmc = MCMC(nuts, num_samples=args.num_samples,
		warmup_steps=args.num_warmups, num_chains=args.num_chains)
	mcmc.run(nn_model, *run_args)
	zs = mcmc.get_samples()['z'].detach().cpu().numpy()
	np.savetxt(save_fn, zs)


	# setting(args)
	# nuts = NUTS(program)
	# mcmc = MCMC(nuts, 2000)
	# mcmc.run(nn_model, target)
	# zs = mcmc.get_samples()['z'].detach().cpu().numpy()
	# np.savetxt(save_fn, zs)
