
from __future__ import division, print_function

import torch, pyro, argparse, os
import pyro.distributions as dist
from pyro.infer.mcmc import MCMC, HMC, NUTS
import numpy as np
from multiprocessing import set_start_method

from nn_models import *

def program_ood(nn_model, label, std=0.05):
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

def setting(args):
	folder = 'results/ood/%s'%args.dataset
	os.makedirs(folder, exist_ok=True)
	save_fn = '%s/%s_%i.txt'%(folder, args.nn_model, args.label)
	if (not args.overwrite) and os.path.isfile(save_fn):
		print(save_fn+' already exists!')
		quit()
	if not os.path.isfile(save_fn):
		fo = open(save_fn, 'w') # write the file first to signal working on it.
		fo.write('\n')
		fo.close()
	if args.dataset == 'fashion_mnist':
		classifier_path = 'saved_models/fashion_mnist_classifier_23569.pth'
		vae_path = 'saved_models/fashion_mnist_vae_01478.pth'
		gan_path = 'saved_models/fashion_mnist_gan_01478.pth'
	elif args.dataset == 'mnist':
		classifier_path = 'saved_models/mnist_classifier_01369.pth'
		vae_path = 'saved_models/mnist_vae_24578.pth'
		gan_path = 'saved_models/mnist_gan_24578.pth'

	if args.nn_model == 'vae':
		raise Exception('vae currently not supported')
		nn_model = VAEModel(args.dataset, vae_path=vae_path, classifier_path=classifier_path)
	elif args.nn_model == 'gan':
		nn_model = GANModel(args.dataset, gan_path=gan_path, classifier_path=classifier_path)
	label = args.label
	return nn_model, label, save_fn

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
	parser.add_argument('--label', type=int)
	args = parser.parse_args()
	nn_model, label, save_fn = setting(args)

	nuts = NUTS(program_ood)
	mcmc = MCMC(nuts, num_samples=args.num_samples, 
		warmup_steps=args.num_warmups, num_chains=args.num_chains)
	mcmc.run(nn_model, label)
	zs = mcmc.get_samples()['z'].detach().cpu().numpy()
	np.savetxt(save_fn, zs)
