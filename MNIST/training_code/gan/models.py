
from __future__ import division

from torch import nn

class Generator(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(Generator, self).__init__()
		self.leaky_relu = nn.LeakyReLU(0.2)
		self.tanh = nn.Tanh()
		self.linear1 = nn.Linear(input_dim, 256)
		self.linear2 = nn.Linear(256, 512)
		self.linear3 = nn.Linear(512, 1024)
		self.linear4 = nn.Linear(1024, output_dim)

	def forward(self, zs):
		output = zs
		output = self.leaky_relu(self.linear1(output))
		output = self.leaky_relu(self.linear2(output))
		output = self.leaky_relu(self.linear3(output))
		output = self.tanh(self.linear4(output))
		return output

class Discriminator(nn.Module):
	def __init__(self, input_dim): # output_dim is 2
		super(Discriminator, self).__init__()
		self.leaky_relu = nn.LeakyReLU(0.2)
		self.dropout = nn.Dropout(0.3)
		self.sigmoid = nn.Sigmoid()
		self.linear1 = nn.Linear(input_dim, 1024)
		self.linear2 = nn.Linear(1024, 512)
		self.linear3 = nn.Linear(512, 256)
		self.linear4 = nn.Linear(256, 1)

	def forward(self, imgs):
		output = imgs
		output = self.dropout(self.leaky_relu(self.linear1(output)))
		output = self.dropout(self.leaky_relu(self.linear2(output)))
		output = self.dropout(self.leaky_relu(self.linear3(output)))
		output = self.sigmoid(self.linear4(output))
		return output
