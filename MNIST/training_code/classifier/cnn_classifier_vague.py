from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 1)
		self.dropout1 = nn.Dropout2d(0.25)
		self.dropout2 = nn.Dropout2d(0.5)
		self.fc1 = nn.Linear(9216, 128)
		self.fc2 = nn.Linear(128, 10)

	def forward(self, x):
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
		output = F.log_softmax(x, dim=1)
		return output

def train_epoch_ambiguous(model, train_loader, optimizer, conf, device):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		target_np = target.detach().cpu().numpy()
		target_prob = np.zeros((len(target), 10))
		target_prob[range(len(target)), target_np] = conf
		target_prob[range(len(target)), (target_np+1)%10] = 1 - conf
		loss = F.kl_div(output, torch.tensor(target_prob).float().to(device), reduction='batchmean')
		loss.backward()
		optimizer.step()

def test(model, test_loader, device):
	model.eval()
	test_loss = 0
	correct = 0
	outputs = []
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			outputs.append(output.detach().cpu().numpy())
			test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()
	outputs = np.exp(np.vstack(outputs))
	outputs.sort(axis=1)
	avg_probs = np.mean(outputs, axis=0)
	print('Test set: Avg prob: '+', '.join(['%0.2f'%p for p in avg_probs]))
	test_loss /= len(test_loader.dataset)

	print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))

def train_ambiguous(dataset_name, conf, n_epochs=100, batch_size=128, device='cuda'):

	folder = 'models/%s_ambiguous_%0.1f'%(dataset_name, conf)
	# if os.path.isdir(folder):
	# 	print('%s already exist. Skipping...'%folder)
	# 	return
	os.makedirs(folder, exist_ok=True)

	dset_class = {'fashion_mnist': FashionMNIST, 'mnist': MNIST}[dataset_name]
	train_dset = dset_class('./data', train=True, download=True, transform=ToTensor())
	test_dset = dset_class('./data', train=False, transform=ToTensor())

	train_loader = torch.utils.data.DataLoader(train_dset,
		batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_dset, 
		batch_size=batch_size, shuffle=True)

	model = Net().to(device)
	optimizer = optim.Adam(model.parameters())

	for epoch in range(1, n_epochs+1):
		train_epoch_ambiguous(model, train_loader, optimizer, conf, device)
		test(model, test_loader, device)
		torch.save(model.state_dict(), "%s/cnn_%i.pt"%(folder, epoch))

if __name__ == '__main__':
	
	train_ambiguous('mnist', conf=0.5)
