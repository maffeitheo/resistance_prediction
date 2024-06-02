import torch.nn as nn
import torch

class CNN(nn.Module):

	def __init__(self, number_of_loci, drug):
		super().__init__()


		#Conv2D with 64 filters, 5 by 5 kernel size
		self.conv2d = nn.Conv2d(number_of_loci, 64, (5,12))
		self.conv1d_1 = nn.Conv2d(64,64, (1,12))
		self.conv1d_2 = nn.Conv2d(64,32, (1,3))
		self.conv1d_3 = nn.Conv2d(32,32, (1,3))

		self.maxpool = nn.MaxPool1d(3)
		self.activation = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.fc1 = nn.Linear(26720,256, bias = True)
		self.fc2 = nn.Linear(256,256, bias = True)
		self.fc3 = nn.Linear(256,1, bias = True)


	def forward(self, x):

		#import pdb; pdb.set_trace()
		x = x.permute((0,1,3,2))
		#Initial Convolution Block
		x = self.activation(self.conv2d(x))


		#Main Convolutional Block
		x = self.activation(self.conv1d_1(x))
		#x = self.activation(x)
		x = self.maxpool(x.squeeze(2)).unsqueeze(2)
		x = self.activation(self.conv1d_2(x))
		x = self.activation(self.conv1d_3(x))
		x = self.maxpool(x.squeeze(2)).unsqueeze(2)
		x = torch.flatten(x, start_dim=1)


		#Output Block
		x = self.fc1(x)
		x = self.activation(x)
		x = self.fc2(x)
		x = self.activation(x)
		x = self.fc3(x)
		x = self.sigmoid(x)


		return x
	


