"Here we will define our generator network"

import torch.nn as nn
import torch
class Generator(nn.Module):
	def __init__(self, noise_dim,output_shape = 64,output_dim=3):
		super(Generator, self).__init__()
		self.network = nn.Sequential(
			# first bloc
			nn.ConvTranspose2d(noise_dim, 512, 4, 1, 0, bias=False),    # noise input and values taken from the official paper
            nn.BatchNorm2d(512),
            nn.ReLU(True),

			## second bloc
			nn.ConvTranspose2d(512,256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

			## third bloc
			nn.ConvTranspose2d(256,128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

			## Fourth bloc
			nn.ConvTranspose2d(128, output_shape ,4,2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

			# last bloc
			nn.ConvTranspose2d(64, output_dim, 4,2,1, bias=False),
			nn.Tanh()
		)


	def forward(self, x):
         return self.network(x)


"Quick visualisation of our generator "

if __name__ =='__main__': 
	device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
	gen = Generator(100).to(device)
	print(gen)


