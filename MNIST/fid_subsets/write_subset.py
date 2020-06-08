import torch, torchvision, pyro, argparse, os
import numpy as np
from multiprocessing import set_start_method
import matplotlib.pyplot as plt
from nn_models import *
import os

def random_latent(latent_dim):

    loc = torch.zeros(latent_dim)
    cov = torch.eye(latent_dim)
    latent_space = torch.distributions.MultivariateNormal(loc, cov)
    return latent_space.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nn-model', default='vae')
    parser.add_argument('--nn-model-path', default='')
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--save-dir')
    args = parser.parse_args()

    if args.nn_model == 'vae':
        nn_model = VAEModel(args.dataset, vae_path=args.nn_model_path)
    elif args.nn_model == 'gan':
        nn_model = GANModel(args.dataset, gan_path=args.nn_model_path)

    if nn_model.device == 'cuda':
        typ = torch.cuda.FloatTensor
    elif nn_model.device == 'cpu':
        typ = torch.FloatTensor
    torch.set_default_tensor_type(typ)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    latent_dim = nn_model.latent_dim


    for i in range(0, 1000):
        latent = random_latent(latent_dim)
        img = nn_model.generate(torch.tensor(latent))
        img = img.reshape(28, 28)
        print (img.shape)
        print (type(img))
        torchvision.utils.save_image(img, args.save_dir + str(i) + '.jpg')
        # plt.imshow(img.cpu().detach().numpy(), cmap='gray')
        # plt.show()
