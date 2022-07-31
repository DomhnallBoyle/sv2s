import torch

BATCH_SIZE = 32
TIMESTEPS = 25
HEIGHT = 88
WIDTH = 88


class Encoder(torch.nn.Module):

	# https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/blob/master/espnet/nets/pytorch_backend/backbones/conv3d_extractor.py
	
	def __init__(self):
		super().__init__()
		
		self.conv3d_1 = torch.nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
		self.maxpool3d_1 = torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

		self.conv2d_1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.conv2d_2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
		# self.conv2d_3 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3))
		# self.conv2d_4 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3))
	
		# self.avpool2d_1 = torch.nn.AvgPool2d()	

	def forward(self, x):
		x = self.conv3d_1(x)
		x = self.maxpool3d_1(x)	
		x = x.reshape((BATCH_SIZE * TIMESTEPS, x.shape[1], *x.shape[-2:]))
		
		for _ in range(2): 
			x = self.conv2d_1(x)
			x = self.conv2d_1(x)
		for _ in range(2): 
			print(x.shape)
			x = self.conv2d_2(x)
		print(x.shape)
	
		return x


class Decoder(torch.nn.Module):
	
	def __init__(self):
		super().__init__()

	def forward(self, x):
		pass


def main():
	encoder = Encoder()	
	x = torch.rand((BATCH_SIZE, 1, TIMESTEPS, HEIGHT, WIDTH))
	x = encoder(x)


if __name__ == '__main__': 
	main()

