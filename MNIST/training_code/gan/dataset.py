
from __future__ import division

import torch, torchvision

from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


def MNIST(batch_size):
	train = torchvision.datasets.MNIST('data/', train=True, download=True)
	test = torchvision.datasets.MNIST('data/', train=False, download=True)

	train_X = (np.vstack([np.array(t[0]).flatten() for t in train]).astype('float32') / 255 - 0.5) / 0.5
	train_y = np.array([t[1] for t in train]).astype('int64')

	test_X = (np.vstack([np.array(t[0]).flatten() for t in test]).astype('float32') / 255 - 0.5 ) / 0.5
	test_y = np.array([t[1] for t in test]).astype('int64')

	# ss = StandardScaler()
	# combined = np.vstack([train_X, test_X])
	# combined = ss.fit_transform(combined)	
	# train_X = combined[:len(train_X)]
	# test_X = combined[len(train_X):]

	train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_X), torch.tensor(train_y))
	test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_X), torch.tensor(test_y))

	train_loader = torch.utils.data.DataLoader(train_dataset, 
		batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
	test_loader = torch.utils.data.DataLoader(test_dataset, 
		batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
	return train_loader, test_loader
