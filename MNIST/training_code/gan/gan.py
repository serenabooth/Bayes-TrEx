
from __future__ import division

import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import itertools

from dataset import MNIST
from models import Generator, Discriminator

max_epoch = 150
batch_size = 128
z_dim = 100
train_loader, test_loader = MNIST(128)

G = Generator(z_dim, 28*28).cuda()
D = Discriminator(28*28).cuda()
G_opt = Adam(G.parameters(), lr=0.0002)
D_opt = Adam(D.parameters(), lr=0.0002)
loss_func = nn.BCELoss().cuda()

zs_fixed = torch.randn((10 * 10, 100))

def plot_results():
	zs = torch.randn((10 * 10, 100)).cuda()
	gen_imgs = G(zs)
	fig, ax = plt.subplots(10, 10)
	for i, j in itertools.product(range(10), range(10)):
		ax[i, j].get_xaxis().set_visible(False)
		ax[i, j].get_yaxis().set_visible(False)
	for k in range(10*10):
		i = k // 10
		j = k % 10
		ax[i, j].cla()
		ax[i, j].imshow(gen_imgs[k, :].cpu().data.view(28, 28).numpy(), cmap='gray')
	plt.show()

for epoch in xrange(max_epoch):
	for real_imgs, _ in train_loader:
		# train D
		D.zero_grad()
		zs = torch.randn((batch_size, z_dim)).cuda()
		fake_imgs = G(zs)
		D_fake_preds = D(fake_imgs).view(-1)
		fake_labels = torch.tensor([0.]*batch_size).cuda()
		D_fake_loss = loss_func(D_fake_preds, fake_labels)

		real_imgs = real_imgs.cuda()
		D_real_preds = D(real_imgs).view(-1)
		real_labels = torch.tensor([1.]*batch_size).cuda()
		D_real_loss = loss_func(D_real_preds, real_labels)

		D_loss = D_fake_loss + D_real_loss
		D_loss.backward()
		D_opt.step()
		
		# train G
		G.zero_grad()
		zs = torch.randn((batch_size, z_dim)).cuda()
		fake_imgs = G(zs)
		D_fake_preds = D(fake_imgs).view(-1)
		inverted_fake_labels = torch.tensor([1.]*batch_size).cuda()
		G_loss = loss_func(D_fake_preds, inverted_fake_labels)
		G_loss.backward()
		G_opt.step()

	print 'epoch:', epoch+1, 'D-loss:', '%.2f'%D_loss.detach().cpu().item(), 'G-loss:', '%.2f'%G_loss.detach().cpu().item()
	# test discriminator
	overall_confs = []
	if epoch > 100:
		digit_confs = [[] for _ in xrange(10)]
		for test_imgs, test_labels in test_loader:
			test_imgs = test_imgs.cuda()
			likelihood = D(test_imgs).view(-1)
			for li, label in zip(likelihood.detach().cpu().numpy(), test_labels.detach().cpu().numpy()):
				digit_confs[label].append(li)
		confs = [np.mean(conf) for conf in digit_confs]
		overall_confs.append(confs)
print np.mean(overall_confs, axis=0)
