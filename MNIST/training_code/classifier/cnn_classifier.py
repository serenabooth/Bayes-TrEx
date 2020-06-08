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


class SelectLabelDataset(Dataset):
	def __init__(self, base_dset_class, keep_labels, root, train=True, download=False, 
		transform=None, target_transform=None):

		keep_labels = set(keep_labels)
		self.keep_labels = keep_labels
		base_dset = base_dset_class(root, train=train, download=download, 
			transform=transform, target_transform=target_transform)
		self.xs = []
		self.ys = []
		for i in xrange(len(base_dset)):
			x, y = base_dset[i]
			if y in keep_labels:
				self.xs.append(x)
				self.ys.append(y)
	def __len__(self):
		return len(self.xs)
	def __getitem__(self, idx):
		return self.xs[idx], self.ys[idx]

def train_epoch(model, device, train_loader, optimizer, epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()


def test(model, device, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(test_loader.dataset)

	print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))

def train(dataset_name, n_epochs=100, batch_size=128, device='cuda'):

	folder = 'models/%s'%dataset_name
	if os.path.isdir(folder):
		print('%s already exist. Skipping...'%folder)
		return
	os.makedirs(folder)

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
		train_epoch(model, device, train_loader, optimizer, epoch)
		test(model, device, test_loader)
		torch.save(model.state_dict(), "%s/cnn_%i.pt"%(folder, epoch))

def train_select_labels(dataset_name, labels, n_epochs=100, batch_size=128, device='cuda'):

	folder = 'models/%s_w_%s'%(dataset_name, ''.join(map(str, labels)))
	if os.path.isdir(folder):
		print('%s already exist. Skipping...'%folder)
		return
	os.makedirs(folder)

	dset_class = {'fashion_mnist': FashionMNIST, 'mnist': MNIST}[dataset_name]
	train_dset = SelectLabelDataset(dset_class, labels, './data', train=True, download=True, transform=ToTensor())
	test_dset = SelectLabelDataset(dset_class, labels, './data', train=False, transform=ToTensor())

	train_loader = torch.utils.data.DataLoader(train_dset,
		batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_dset, 
		batch_size=batch_size, shuffle=True)

	model = Net().to(device)
	optimizer = optim.Adam(model.parameters())

	for epoch in range(1, n_epochs+1):
		train_epoch(model, device, train_loader, optimizer, epoch)
		test(model, device, test_loader)
		torch.save(model.state_dict(), "%s/cnn_%i.pt"%(folder, epoch))


if __name__ == '__main__':
	
	# train('fashion_mnist')
	train_select_labels('mnist', [0, 1, 3, 6, 9])
