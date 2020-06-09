import torch, torchvision, pyro, argparse, os
import numpy as np
from multiprocessing import set_start_method
import matplotlib.pyplot as plt
from nn_models import *
import os, csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nn-model')
    parser.add_argument('--nn-model-path')
    parser.add_argument('--dataset')
    parser.add_argument('--latent-file')
    parser.add_argument('--render', default=True)

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

    latent_dim = nn_model.latent_dim

    latents_file = args.latent_file
    save_dir = args.latent_file.replace('.txt', '')
    # if the save directory doesn't exist, create it.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    in_txt_reader = csv.reader(open(latents_file, "rt"), delimiter = " ")
    for i, row in enumerate(in_txt_reader):
        row = [float(val) for val in row]
        img = nn_model.generate(torch.tensor(row))
        img = img.reshape(28, 28)
        if args.render:
            plt.imshow(img.cpu().detach().numpy(), cmap='gray')
            plt.show()
        torchvision.utils.save_image(img, save_dir + "/" + str(i) + '.jpg')
