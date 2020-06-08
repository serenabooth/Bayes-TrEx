
from __future__ import division, print_function

import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['MNISTClassifier', 'MNISTVAE', 'MNISTGAN', 
		   'FashionMNISTClassifier', 'FashionMNISTVAE', 'FashionMNISTGAN', 
		   'LeNet']

class Flatten(nn.Module):
	def forward(self, x):
		return x.reshape(x.size(0), -1)

class LeNet(nn.Module):
	def __init__(self, temperature=1.):
		super(LeNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 20, 5)
		self.relu1 = nn.ReLU()
		self.maxpool1 = nn.MaxPool2d(2)
		self.conv2 = nn.Conv2d(20, 50, 5)
		self.relu2 = nn.ReLU()
		self.maxpool2 = nn.MaxPool2d(2)
		self.flatten = Flatten()
		self.fc3 = nn.Linear(800, 500)
		self.relu3 = nn.ReLU()
		self.fc4 = nn.Linear(500, 10)
		self.temperature = temperature

	def forward(self, x):
		x = x.reshape(-1, 1, 28, 28)
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.maxpool2(x)
		x = x.permute(0, 2, 3, 1)
		x = self.flatten(x)
		x = self.fc3(x)
		x = self.relu3(x)
		x = self.fc4(x)
		return F.softmax(x / self.temperature, dim=1)

class MNISTClassifier(nn.Module):
	def __init__(self, temperature=1.):
		super(MNISTClassifier, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 1)
		self.dropout1 = nn.Dropout2d(0.25)
		self.dropout2 = nn.Dropout2d(0.5)
		self.fc1 = nn.Linear(9216, 128)
		self.fc2 = nn.Linear(128, 10)
		self.temperature = temperature

	def forward(self, x):
		x = x.reshape(-1, 1, 28, 28)
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.max_pool2d(x, 2)
		x = self.dropout1(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout2(x)
		x = self.fc2(x)
		return F.softmax(x / self.temperature, dim=1)

class MNISTVAE(nn.Module):
	def __init__(self, latent_dim):
		super(MNISTVAE, self).__init__()

		self.fc1 = nn.Linear(784, 400)
		self.fc21 = nn.Linear(400, latent_dim)
		self.fc22 = nn.Linear(400, latent_dim)
		self.fc3 = nn.Linear(latent_dim, 400)
		self.fc4 = nn.Linear(400, 784)

	def decode(self, z):
		h3 = F.relu(self.fc3(z))
		return torch.sigmoid(self.fc4(h3))

class MNISTGAN(nn.Module):
	def __init__(self, latent_dim):
		super(MNISTGAN, self).__init__()
		self.main = nn.Sequential(
			# input is Z, going into a convolution
			nn.ConvTranspose2d(latent_dim, 64 * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(64 * 8),
			nn.ReLU(True),
			# state size. (64*8) x 4 x 4
			nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(64 * 4),
			nn.ReLU(True),
			# state size. (64*4) x 8 x 8
			nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(64 * 2),
			nn.ReLU(True),
			# state size. (64*2) x 16 x 16
			nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.ConvTranspose2d(64, 1, kernel_size=1, stride=1, padding=2, bias=False),
			nn.Sigmoid()
		)
		self.latent_dim = latent_dim

	def forward(self, latent):
		latent = latent.reshape(-1, self.latent_dim, 1, 1)
		output = self.main(latent)
		return output[:, 0, :, :]

FashionMNISTGAN = MNISTGAN
FashionMNISTVAE = MNISTVAE
FashionMNISTClassifier = MNISTClassifier
