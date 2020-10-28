import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data as data_utils
# from torchsummary import summary
# from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split

import torch.nn.init as init

from six.moves import range
from numba import njit, prange
import numpy as np
import scipy as sp
from scipy import spatial
from scipy.special import softmax
import random
import scipy.io
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import pickle as pk
# from gurobipy import *
from time import time

from models import *
from init_conditions import *
from utils import *
from al_algorithms import *

import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
identificationTag = ['output', 'kcd-and-output']

activation = {}
def get_activation(name):
	global activation
	def hook(model, input, output):
		# global activation
		activation[name] = output.detach()
	# print('gg=', /activation)
	return hook


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class VAE_forMNIST(nn.Module):
	"""Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
	def __init__(self, z_dim=32, nc=1):
		super(VAE_forMNIST, self).__init__()
		self.z_dim = z_dim
		self.nc = nc
		# self.encoder = nn.Sequential(
		# 	nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
		# 	nn.BatchNorm2d(128),
		# 	nn.ReLU(True),
		# 	View((-1, 128*14*14)),       # ((28 + 2) - 4)/2 + 1                          # B, 1024*4*4
		# )

		# self.fc_mu = nn.Linear(128*14*14, z_dim)                            # B, z_dim
		# self.fc_logvar = nn.Linear(128*14*14, z_dim)                            # B, z_dim
		# self.decoder = nn.Sequential(
		# 	nn.Linear(z_dim, 128*28*28),                           # B, 1024*8*8
		# 	View((-1, 128, 28, 28)),                               # B, 1024,  8,  8
		# 	# nn.ConvTranspose2d(128, 512, 4, 2, 1, bias=False),   # B,  512, 16, 16
		# 	# nn.BatchNorm2d(512),
		# 	# nn.ReLU(True),
		# 	# nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
		# 	# nn.BatchNorm2d(256),
		# 	# nn.ReLU(True),
		# 	# nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
		# 	# nn.BatchNorm2d(128),
		# 	# nn.ReLU(True),
		# 	nn.ConvTranspose2d(128, nc, 1),                       # B,   nc, 64, 64
		# )
		self.encoder = nn.Sequential(
			nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
			# nn.BatchNorm2d(128),
			nn.Dropout2d(0.25),
			nn.ReLU(True),
			nn.Conv2d(128, 256, 4, 2, 1, bias=False),
			# nn.BatchNorm2d(256),
			nn.Dropout2d(0.25),
			nn.ReLU(True),
			View((-1, 256*7*7)),       # ((28 + 2) - 4)/2 + 1                          # B, 1024*4*4
		)

		self.fc_mu = nn.Linear(256*7*7, z_dim)                            # B, z_dim
		self.fc_logvar = nn.Linear(256*7*7, z_dim)                            # B, z_dim
		self.decoder = nn.Sequential(
			nn.Linear(z_dim, 256*14*14),                           # B, 1024*8*8
			View((-1, 256, 14, 14)),                               # B, 1024,  8,  8
			nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),   # B,  512, 16, 16
			# nn.BatchNorm2d(128),
			nn.Dropout2d(0.25),
			nn.ReLU(True),
			# nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
			# nn.BatchNorm2d(128),
			# nn.ReLU(True),
			nn.ConvTranspose2d(128, nc, 1),                       # B,   nc, 64, 64
		)
		self.weight_init()

	def weight_init(self):
		for block in self._modules:
			try:
				for m in self._modules[block]:
					kaiming_init(m)
			except:
				kaiming_init(block)

	def forward(self, x):
		z = self._encode(x)
		mu, logvar = self.fc_mu(z), self.fc_logvar(z)
		z = self.reparameterize(mu, logvar)
		x_recon = self._decode(z)

		# print(x_recon.shape())
		return x_recon, z, mu, logvar

	def reparameterize(self, mu, logvar):
		stds = (0.5 * logvar).exp()
		epsilon = torch.randn(*mu.size())
		if mu.is_cuda:
			stds, epsilon = stds.cuda(), epsilon.cuda()
		latents = epsilon * stds + mu
		return latents

	def _encode(self, x):
		return self.encoder(x)

	def _decode(self, z):
		return self.decoder(z)


class Discriminator_forMNIST(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10):
        super(Discriminator_forMNIST, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        # init.kaiming_normal(m.weight)
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


def getModelForMNIST_VAAL(numTrain, nb_classes, appendNoiseLayer = False,
						 p_yp_given_y = None, noiseTrainable = False):

	# make sure to add 'output' as name for the output layer, 
	# and add name 'kcd' for which coreset
	# output is needed as the delta function, and 'kcd-and-output' for which output 
	# and coreset delta is same

	if appendNoiseLayer:
		model = MNIST_NN_withNoise(p_yp_given_y = p_yp_given_y, noiseTrainable = noiseTrainable)
	else:
		model = MNIST_NN()
	model.to(device)

	lr = 0.001
	decay = 3.5 / float(numTrain)
	if appendNoiseLayer:
		paramsForOptim = [
			{'params':model.convBlock.parameters()},
			{'params':model.linearBlock.parameters()},
			{'params':model.Dense.parameters(), 'weight_decay': decay},
			{'params':model.nl.parameters()}
			]
	else:
		paramsForOptim = [
			{'params':model.convBlock.parameters()},
			{'params':model.linearBlock.parameters()},
			{'params':model.Dense.parameters(), 'weight_decay': decay}
			]
	optimizer = torch.optim.Adam(paramsForOptim, lr =lr)

	vae = VAE_forMNIST(z_dim=32)
	discriminator = Discriminator_forMNIST(z_dim=32)

	vae.to(device)
	discriminator.to(device)
	optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
	optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)

	return model, optimizer, vae, optim_vae, discriminator, optim_discriminator


def getModelForEMNIST_VAAL(numTrain, nb_classes, appendNoiseLayer = False,
						 p_yp_given_y = None, noiseTrainable = False):

	# make sure to add 'output' as name for the output layer, 
	# and add name 'kcd' for which coreset
	# output is needed as the delta function, and 'kcd-and-output' for which output 
	# and coreset delta is same

	if appendNoiseLayer:
		model = EMNIST_NN_withNoise(p_yp_given_y = p_yp_given_y, noiseTrainable = noiseTrainable)
	else:
		model = EMNIST_NN()
	model.to(device)

	lr = 0.001
	decay = 3.5 / float(numTrain)
	if appendNoiseLayer:
		paramsForOptim = [
			{'params':model.convBlock.parameters()},
			{'params':model.linearBlock.parameters()},
			{'params':model.Dense.parameters(), 'weight_decay': decay},
			{'params':model.nl.parameters(), 'weight_decay': 0.5}
			]
	else:
		paramsForOptim = [
			{'params':model.convBlock.parameters()},
			{'params':model.linearBlock.parameters()},
			{'params':model.Dense.parameters(), 'weight_decay': decay}
			]
	optimizer = torch.optim.Adam(paramsForOptim, lr =lr)

	vae = VAE_forMNIST(z_dim=32)
	discriminator = Discriminator_forMNIST(z_dim=32)

	vae.to(device)
	discriminator.to(device)
	optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
	optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)

	return model, optimizer, vae, optim_vae, discriminator, optim_discriminator


def lr_schedule(optimizer, epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 120:
        lr *= 0.5e-3
    elif epoch > 100:
        lr *= 1e-3
    elif epoch > 80:
        lr *= 1e-2
    elif epoch > 60:
        lr *= 1e-1
    print('Learning rate: ', lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def initTorchModelPath(root = '../tempTorch', name = 'currTorchModel,pt'):
	if not os.path.exists(root):
		os.mkdir(root)

	return os.path.join(root, name)

class EarlyStopping(object):

	def __init__(self, patience = 0, monitor = np.inf, PATH = './currTorchModel.pt', verbose=False):
		self.totalPatience = patience
		self.patience = patience
		self.monitor = monitor
		self.PATH = PATH
		self.verbose = verbose

	def check(self, model, currVal):
		if currVal < self.monitor:
			self.monitor = currVal
			self.patience = self.totalPatience
			torch.save(model.state_dict(), self.PATH)
			if self.verbose:
				print('resetting early stop counter...')
		else:
			self.patience -= 1
			if self.verbose:
				print('early stopping counter begins: patience remaining = %d'%(self.patience))
		if self.patience < 0:		
			return True
		else:
			return False


class EarlyStopping1(object):

	def __init__(self, patience = 0, monitor = np.inf, PATH = './currTorchModel.pt', verbose=False):
		self.totalPatience = patience
		self.patience = patience
		self.monitor1 = monitor
		self.monitor2 = 0
		self.PATH = PATH
		self.verbose = verbose

	def check(self, model, valLoss, valAcc):
		if valLoss < self.monitor1 or valAcc > self.monitor2:
			self.monitor1 = np.min([self.monitor1, valLoss])
			self.monitor2 = np.max([self.monitor2, valAcc])
			self.patience = self.totalPatience
			torch.save(model.state_dict(), self.PATH)
			if self.verbose:
				print('resetting early stop counter...')
		else:
			self.patience -= 1
			if self.verbose:
				print('early stopping counter begins: patience remaining = %d'%(self.patience))
		if self.patience < 0:		
			return True
		else:
			return False


class SaveBestModel(object):

	def __init__(self, monitor = np.inf, PATH = './currTorchModel.pt', 
					verbose=False):
		
		self.monitor = monitor
		self.PATH = PATH
		self.verbose = verbose

	def check(self, model, currVal, comp='min'):
		if comp is 'min':
			if currVal < self.monitor:
				self.monitor = currVal
				torch.save(model.state_dict(), self.PATH)
				if self.verbose:
					print('saving best model...')
		elif comp is 'max':
			if currVal > self.monitor:
				self.monitor = currVal
				torch.save(model.state_dict(), self.PATH)
				if self.verbose:
					print('saving best model...')


def lr_schedule_SGD(optimizer, epoch):
	"""Learning Rate Schedule
	Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
	Called automatically every epoch as part of callbacks during training.
	# Arguments
		epoch (int): The number of epochs
	# Returns
		lr (float32): learning rate
	"""

	lr_old = optimizer.param_groups[0]['lr']

	# lr_new = lr_old * 

	# lr = 0.05 * (0.5 **(epoch//30))
	# lr = 0.05 * (0.5 **(epoch//20))
	# lr = 0.001 * (0.1 **(epoch//20))

	# if epoch > 20:
	# 	lr_new = lr_old * 1e-1
	# elif epoch > 40:
	# 	lr_new = lr_old * 1e-2
	# elif epoch > 60:
	# 	lr_new = lr_old * 1e-3
	# else:
	# 	lr_new = lr_old
	if epoch % 20 == 0:
		lr_new = lr_old * 0.1
	else:
		lr_new = lr_old

	# print('Changing Learning rate to: %f'%(lr_new))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr_new


class VAE_forCIFAR(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=32, nc=3):
        super(VAE_forCIFAR, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
            # nn.BatchNorm2d(128),
			nn.Dropout2d(0.25),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16
            # nn.BatchNorm2d(256),
			nn.Dropout2d(0.25),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8
            # nn.BatchNorm2d(512),
			nn.Dropout2d(0.25),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),            # B, 1024,  4,  4
            # nn.BatchNorm2d(1024),
			nn.Dropout2d(0.25),
            nn.ReLU(True),
            View((-1, 1024*2*2)),                                 # B, 1024*4*4
        )

        self.fc_mu = nn.Linear(1024*2*2, z_dim)                            # B, z_dim
        self.fc_logvar = nn.Linear(1024*2*2, z_dim)                            # B, z_dim
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024*4*4),                           # B, 1024*8*8
            View((-1, 1024, 4, 4)),                               # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),   # B,  512, 16, 16
            # nn.BatchNorm2d(512),
			nn.Dropout2d(0.25),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
            # nn.BatchNorm2d(256),
			nn.Dropout2d(0.25),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
            # nn.BatchNorm2d(128),
			nn.Dropout2d(0.25),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1),                       # B,   nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x):
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class Discriminator_forCIFAR(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10):
        super(Discriminator_forCIFAR, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)


def getCifar10Model_VAAL(numTrain, nb_classes, appendNoiseLayer = False,
						 p_yp_given_y = None, noiseTrainable = False):

	# make sure to add 'output' as name for the output layer, 
	# and add name 'kcd' for which coreset
	# output is needed as the delta function, and 'kcd-and-output' for which output 
	# and coreset delta is same
	cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

	cfg1 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
	cfg_1p = ['BN', '', '', 'BN', '', '', 'BN', '', '', '', 'BN', '', '', '', 'BN', '', '', '', '']
	cgf_1_use = zip(cfg1, cfg_1p)

	lr = 0.001
	decay = 2.5 / float(numTrain)
	if appendNoiseLayer:
		model = CIFAR_NN_SimpleNext_withNoise(p_yp_given_y = p_yp_given_y, noiseTrainable = noiseTrainable)
		paramsForOptim = [
			{'params':model.convBlock.parameters()},
			{'params':model.linearBlock.parameters()},
			{'params':model.Dense.parameters(), 'weight_decay': decay},
			{'params':model.nl.parameters(), 'weight_decay': 2.5}
			]
		optimizer = torch.optim.Adam(paramsForOptim, lr =lr)
	else:
		model = CIFAR_NN_SimpleNext()
		paramsForOptim = [
			{'params':model.convBlock.parameters()},
			{'params':model.linearBlock.parameters()},
			{'params':model.Dense.parameters(), 'weight_decay': decay}
			]
		optimizer = torch.optim.Adam(paramsForOptim, lr =lr)
		# model = VGG16_use()
		# optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
	model.to(device)

	vae = VAE_forCIFAR()
	discriminator = Discriminator_forCIFAR(z_dim=32)

	vae.to(device)
	discriminator.to(device)
	optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
	optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)

	return model, optimizer, vae, optim_vae, discriminator, optim_discriminator


def getCifar100Model_VAAL(numTrain, nb_classes, appendNoiseLayer = False,
						 p_yp_given_y = None, noiseTrainable = False):

	# make sure to add 'output' as name for the output layer, 
	# and add name 'kcd' for which coreset
	# output is needed as the delta function, and 'kcd-and-output' for which output 
	# and coreset delta is same
	cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

	cfg1 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
	cfg_1p = ['BN', '', '', 'BN', '', '', 'BN', '', '', '', 'BN', '', '', '', 'BN', '', '', '', '']
	cgf_1_use = zip(cfg1, cfg_1p)

	lr = 0.001
	decay = 2.5 / float(numTrain)
	if appendNoiseLayer:
		model = CIFAR100_NN_SimpleNext_withNoise(p_yp_given_y = p_yp_given_y, noiseTrainable = noiseTrainable)
		paramsForOptim = [
			{'params':model.convBlock.parameters()},
			{'params':model.linearBlock.parameters()},
			{'params':model.Dense.parameters(), 'weight_decay': decay},
			{'params':model.nl.parameters(), 'weight_decay': 2.5}
			]
		optimizer = torch.optim.Adam(paramsForOptim, lr =lr)
	else:
		model = CIFAR100_NN_SimpleNext()
		paramsForOptim = [
			{'params':model.convBlock.parameters()},
			{'params':model.linearBlock.parameters()},
			{'params':model.Dense.parameters(), 'weight_decay': decay}
			]
		optimizer = torch.optim.Adam(paramsForOptim, lr =lr)
		# model = VGG16_use()
		# optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
	model.to(device)

	# summary(model, (3, 32, 32))

	
	vae = VAE_forCIFAR()
	discriminator = Discriminator_forCIFAR(z_dim=32)

	vae.to(device)
	discriminator.to(device)
	optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
	optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)

	return model, optimizer, vae, optim_vae, discriminator, optim_discriminator
	# return model, optimizer


def getSVHNModel_VAAL(numTrain, nb_classes, appendNoiseLayer = False,
						 p_yp_given_y = None, noiseTrainable = False):

	# make sure to add 'output' as name for the output layer, 
	# and add name 'kcd' for which coreset
	# output is needed as the delta function, and 'kcd-and-output' for which output 
	# and coreset delta is same
	cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

	cfg1 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
	cfg_1p = ['BN', '', '', 'BN', '', '', 'BN', '', '', '', 'BN', '', '', '', 'BN', '', '', '', '']
	cgf_1_use = zip(cfg1, cfg_1p)

	lr = 0.001
	decay = 2.5 / float(numTrain)
	if appendNoiseLayer:
		model = SVHN_NN_SimpleNext_withNoise(p_yp_given_y = p_yp_given_y, noiseTrainable = noiseTrainable)
		paramsForOptim = [
			{'params':model.convBlock.parameters()},
			{'params':model.linearBlock.parameters()},
			{'params':model.Dense.parameters(), 'weight_decay': decay},
			{'params':model.nl.parameters(), 'weight_decay': 0.5}
			]
		optimizer = torch.optim.Adam(paramsForOptim, lr =lr)
	else:
		model = SVHN_NN_SimpleNext()
		paramsForOptim = [
			{'params':model.convBlock.parameters()},
			{'params':model.linearBlock.parameters()},
			{'params':model.Dense.parameters(), 'weight_decay': decay}
			]
		optimizer = torch.optim.Adam(paramsForOptim, lr =lr)

	model.to(device)

	# summary(model, (3, 32, 32))

	
	vae = VAE_forCIFAR()
	discriminator = Discriminator_forCIFAR(z_dim=32)

	vae.to(device)
	discriminator.to(device)
	optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
	optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)

	return model, optimizer, vae, optim_vae, discriminator, optim_discriminator
	# return model, optimizer


def read_data(dataloader, labels=True):
	if labels:
		while True:
			for img, label in dataloader:
				yield img, label
	else:
		while True:
			for img, _ in dataloader:
				yield img


def vae_loss(x, recon, mu, logvar, beta):
	mse_loss = nn.MSELoss()
	MSE = mse_loss(recon, x)
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	KLD = KLD * beta
	return MSE + KLD


def train_forVAAL(model, vae, discriminator, train_loader, pool_loader, optimizer, optim_vae, optim_discriminator, epoch, verbose = 0,
			 withNoise = False, lr_schedule = None, weight = None):
	"""Training"""
	if lr_schedule is not None:
		optimizer = lr_schedule(optimizer, epoch)
	
	num_vae_steps = 1 # to be consistent with training schemes of other algorithms, and 
	# we will go over the entire samples anyways to have an epoch
	num_adv_steps = 1
	beta = 1
	adversary_param = 1

	# optim_vae = optim.Adam(vae.parameters(), lr=1e-3)
	# optimizer = optim.Adam(model.parameters(), lr=1e-3)
	# optim_discriminator = optim.Adam(discriminator.parameters(), lr=1e-3)

	# optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
	# optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)

	model.train()
	vae.train()
	discriminator.train()
	if withNoise:
		probabilityConstr = ProbabilitySimplex()

	unlabeled_data = read_data(pool_loader, labels=False)

	for batch_idx, (data, target) in enumerate(train_loader):
		# if batch_idx == len(train_loader)-1 and len(train_loader) > 1:
		# 	continue
			# skipping last batch due to possible size mismatch b/w train and pool batch

		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()

		if withNoise:
			output, out_orig, _ = model(data)
			loss = F.nll_loss(output, target)
		else:
			output, _ = model(data)
			criteria = nn.CrossEntropyLoss().cuda()
			loss = criteria(output, target)
		
		# lambda_l2 = 3.5 / 60000
		# all_linear1_params = torch.cat([x.view(-1) for x in model.fc1.parameters()])
		# l2_regularization = torch.norm(all_linear1_params, 2) 
		# loss = criteria(output, target) + lambda_l2 * l2_regularization
		### decay already applied in the optimizer design
		
		loss.backward()
		optimizer.step()
		if withNoise:
			model.nl.apply(probabilityConstr)

		unlabeled_imgs = next(unlabeled_data)
		unlabeled_imgs = unlabeled_imgs.to(device)

		if data.size(0) < unlabeled_imgs.size(0):
			unlabeled_imgs = unlabeled_imgs[:data.size(0),:,:,:]
		# vae step

		for count in range(num_vae_steps):
			recon, z, mu, logvar = vae(data)
			# print(recon.size())
			unsup_loss = vae_loss(data, recon, mu, logvar, beta)
			unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
			transductive_loss = vae_loss(unlabeled_imgs, 
					unlab_recon, unlab_mu, unlab_logvar, beta)
		
			labeled_preds = discriminator(mu)
			unlabeled_preds = discriminator(unlab_mu)
			
			# print('data size:', data.size(0))
			# print('pred size:', unlabeled_imgs.size(0))
			lab_real_preds = torch.ones(data.size(0))
			unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
				
			# if self.args.cuda:
			lab_real_preds = lab_real_preds.cuda()
			unlab_real_preds = unlab_real_preds.cuda()

			criteria = nn.BCELoss().cuda()

			# print('d pred label:', labeled_preds.shape)
			# print('real label:', lab_real_preds.shape)

			dsc_loss = criteria(labeled_preds, lab_real_preds) + \
					criteria(unlabeled_preds, unlab_real_preds)
			total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss
			optim_vae.zero_grad()
			total_vae_loss.backward()
			optim_vae.step()

			# # sample new batch if needed to train the adversarial network
			# if count < (self.args.num_vae_steps - 1):
			# 	labeled_imgs, _ = next(labeled_data)
			# 	unlabeled_imgs = next(unlabeled_data)

			# 	if self.args.cuda:
			# 		labeled_imgs = labeled_imgs.cuda()
			# 		unlabeled_imgs = unlabeled_imgs.cuda()
			# 		labels = labels.cuda()

			# discriminator step

			for count in range(num_adv_steps):
				with torch.no_grad():
					_, _, mu, _ = vae(data)
					_, _, unlab_mu, _ = vae(unlabeled_imgs)
				
				labeled_preds = discriminator(mu)
				unlabeled_preds = discriminator(unlab_mu)
				
				lab_real_preds = torch.ones(data.size(0))
				unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

				# if self.args.cuda:
				lab_real_preds = lab_real_preds.cuda()
				unlab_fake_preds = unlab_fake_preds.cuda()

				criteria = nn.BCELoss().cuda()
				
				dsc_loss = criteria(labeled_preds, lab_real_preds) + \
						criteria(unlabeled_preds, unlab_fake_preds)

				optim_discriminator.zero_grad()
				dsc_loss.backward()
				optim_discriminator.step()

                # # sample new batch if needed to train the adversarial network
				#  Note: not needed as per the default value of num_adv_steps = 1 from the original paper
				# if count < (num_adv_steps - 1):
				# 	labeled_imgs, _ = next(labeled_data)
				# 	unlabeled_imgs = next(unlabeled_data)

				# 	if self.args.cuda:
				# 		labeled_imgs = labeled_imgs.cuda()
				# 		unlabeled_imgs = unlabeled_imgs.cuda()
				# 		labels = labels.cuda()


	if verbose>0:
		print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, batch_idx * len(data), len(train_loader.dataset),
					100. * batch_idx / len(train_loader), loss.item()))
	return loss.item()


def checkAllSameOutput(model, data_loader, withNoise):
	model.eval()
	outputs = []
	correct = 0
	with torch.no_grad():
		for data, target in data_loader:
			data, target = data.to(device), target.to(device)

			if withNoise:
				_, output, _ = model(data)
			else:
				output, _ = model(data)
			# output, _ = model(data)
			predictionClasses = output.argmax(dim=1, keepdim=True)
			correct += predictionClasses.eq(target.view_as(predictionClasses)).sum().item()
			outputs.extend(predictionClasses.cpu())
	
	outputs = np.array(outputs)
	data_acc = correct / len(data_loader.dataset)
	print(outputs)
	if np.sum(np.abs(np.diff(outputs))) == 0:
		return True, data_acc
	else:
		return False, data_acc


def validateForSVHN(model, val_loader, verbose=0, activationName = None):
	model.eval()

	val_loss = 0
	correct = 0
	outputs = []

	with torch.no_grad():
		for data, target in val_loader:
			data, target = data.to(device), target.to(device)

			if activationName is not None:
				_, output, _ = model(data)
			else:
				output, _ = model(data)
			predictionClasses = output.argmax(dim=1, keepdim=True)
			correct += predictionClasses.eq(target.view_as(predictionClasses)).sum().item()
			outputs.extend(predictionClasses.cpu())
			criteria = nn.CrossEntropyLoss().cuda()
			val_loss += criteria(output, target).sum().item()


	val_loss = val_loss/ len(val_loader)
	val_acc = correct / len(val_loader.dataset)
	outputs = np.array(outputs)
	data_acc = correct / len(val_loader.dataset)
	if np.sum(np.abs(np.diff(outputs))) == 0:
		return val_loss, val_acc, True
	else:
		return val_loss, val_acc, False


def validate(model, val_loader, verbose=0, activationName = None):
    """Testing"""
    model.eval()
    val_loss = 0
    correct = 0

    total_val_loss, total_corrections = evalModel(val_loader, model, verbose = verbose, 
                            stochastic_pass = False, compute_metrics = True, activationName = activationName)

    val_loss = total_val_loss/ len(val_loader) # loss function already averages over batch size
    val_acc = total_corrections / len(val_loader.dataset)
    if verbose>0:
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
        print('{{"metric": "Eval - cross entropy Loss", "value": {}, "epoch": {}}}'.format(
            val_loss, epoch))
        print('{{"metric": "Eval - Accuracy", "value": {}, "epoch": {}}}'.format(
            100. * correct / len(val_loader.dataset), epoch))

    return val_loss, val_acc


def test(model, test_loader, verbose=0, activationName = None):
    """Testing"""
    model.eval()
    test_loss = 0
    correct = 0

    total_test_loss, total_corrections = evalModel(test_loader, model, verbose = verbose, 
                            stochastic_pass = False, compute_metrics = True, activationName = activationName)

    test_loss = total_test_loss/ len(test_loader) # loss function already averages over batch size
    test_acc = total_corrections / len(test_loader.dataset)
    if verbose>0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        print('{{"metric": "Eval - cross entropy Loss", "value": {}, "epoch": {}}}'.format(
            test_loss, epoch))
        print('{{"metric": "Eval - Accuracy", "value": {}, "epoch": {}}}'.format(
            100. * correct / len(test_loader.dataset), epoch))

    return test_loss, test_acc


# @ex.config
def cfg():
	params = {
		'dataType':'MNIST',  # MNIST, CIFAR10
		'trainInitSize': 40,
		'Experiments':5,
		'batch_size':128,
	#use a large number of epochs
		'nb_epoch':50,
	#use a large number of dropout iterations
		'dropout_iterations':100,
		'Queries':200,

		'method': 'BALD', # BALD, BALD-noise, best-BALD, Random, k-center, k-center-greedy, BADGE, maxEnt
		'label_noise': False,
		'noiseChannel': 'qSC', # noisyTypewriter, qSC
        'trainNoiseLayer': False,
		'onp': 0.1, # oracle noise probability
	}

	params['mnp'] = params['onp'] # model noise probability
	return params

	# params['acquisition_iterations'] = int(np.ceil(1000/params['Queries']))

def sanityCheckForTrain(model, data_loader, withNoise=False, acc_thres = 0.1):

	all_same_flag, acc = checkAllSameOutput(model, data_loader, withNoise) 
	if (all_same_flag is False) and acc > acc_thres:
		return True
	else:
		return False


def sanityCheckForTrainSVHN(model, data_loader, activationName = None, acc_thres = 0.1):

	global valid_loss_BASE
	# all_same_flag, acc = checkAllSameOutput(model, data_loader, withNoise) 
	val_loss, val_acc, all_same_flag = validateForSVHN(model, data_loader, activationName = activationName)
	if (all_same_flag is False) and val_acc > acc_thres and val_loss <= valid_loss_BASE:
		return True
	else:
		return False


def trainMNIST_VAALRoutine(train_loader, val_loader, pool_loader,
					nb_epoch, nb_classes, verbose=0,
					trainWithNoise=False, params = {}):

	PATH_curr = initTorchModelPath(root = '../tempTorch', 
									name = 'currTorchModel_MNIST' + np.str(params['exp_no']) + '.pt')
	saveModel = SaveBestModel(PATH = PATH_curr, monitor=-np.inf)
	if trainWithNoise:
		if params['trainNoiseLayer']:
			p_yp_given_y = torch.eye(nb_classes)
		else:
			p_yp_given_y = get_qSC_channel(nb_classes, params['onp'])
		model, optimizer, vae, optim_vae, discriminator, optim_discriminator = getModelForMNIST_VAAL(len(train_loader.dataset), nb_classes,
							appendNoiseLayer = True, p_yp_given_y = p_yp_given_y, noiseTrainable = params['trainNoiseLayer'])
		activationName = 'beforeNoise'
	else:
		model, optimizer, vae, optim_vae, discriminator, optim_discriminator = getModelForMNIST_VAAL(len(train_loader.dataset), nb_classes)
		activationName = None

	for epoch in range(1, nb_epoch + 1):
		train_loss = train_forVAAL(model, vae, discriminator, 
								train_loader, pool_loader, optimizer,
								optim_vae, optim_discriminator,
									epoch, verbose=0, withNoise = trainWithNoise)
		val_loss, val_acc = validate(model, val_loader, activationName=activationName)
		if verbose>0:
			print('val loss = %f, val acc = %f'%(val_loss, val_acc))
		if params['saveModel']:
			saveModel.check(model, val_acc, comp='max')

	if params['saveModel']:
		model, _, _, _, _, _ = getModelForMNIST_VAAL(len(train_loader.dataset), nb_classes, appendNoiseLayer=trainWithNoise)
		model.load_state_dict(torch.load(PATH_curr))

	return model, train_loss


def trainEMNIST_VAALRoutine(train_loader, val_loader, pool_loader,
					nb_epoch, nb_classes, verbose=0,
					trainWithNoise=False, params = {}):

	PATH_curr = initTorchModelPath(root = '../tempTorch', 
									name = 'currTorchModel_EMNIST' + np.str(params['exp_no']) + '.pt')
	saveModel = SaveBestModel(PATH = PATH_curr, monitor=-np.inf)
	if trainWithNoise:
		if params['trainNoiseLayer']:
			p_yp_given_y = torch.eye(nb_classes)
		else:
			p_yp_given_y = get_qSC_channel(nb_classes, params['onp'])
		model, optimizer, vae, optim_vae, discriminator, optim_discriminator = getModelForEMNIST_VAAL(len(train_loader.dataset), nb_classes,
							appendNoiseLayer = True, p_yp_given_y = p_yp_given_y, noiseTrainable = params['trainNoiseLayer'])
		activationName = 'beforeNoise'
	else:
		model, optimizer, vae, optim_vae, discriminator, optim_discriminator = getModelForEMNIST_VAAL(len(train_loader.dataset), nb_classes)
		activationName = None

	for epoch in range(1, nb_epoch + 1):
		train_loss = train_forVAAL(model, vae, discriminator, 
								train_loader, pool_loader, optimizer,
								optim_vae, optim_discriminator,
									epoch, verbose=0, withNoise = trainWithNoise)
		val_loss, val_acc = validate(model, val_loader, activationName=activationName)
		if verbose>0:
			print('val loss = %f, val acc = %f'%(val_loss, val_acc))
		if params['saveModel']:
			saveModel.check(model, val_acc, comp='max')

	if params['saveModel']:
		model, _, _, _, _, _ = getModelForEMNIST_VAAL(len(train_loader.dataset), nb_classes, appendNoiseLayer=trainWithNoise)
		model.load_state_dict(torch.load(PATH_curr))

	return model, train_loss


def trainCifarRoutine(train_loader, val_loader,
						nb_epoch, nb_classes, verbose = 0,
						trainWithNoise=False, params = {}, initialModel = False):

	# schedular = ReduceLROnPlateau(optimizer, 'min', verbose=True, 
			# 								patience = 5, factor = 0.1,
			# 								min_lr = 0.5e-6)
	def prepareCifarModel():

		if trainWithNoise:# adding noise layer
			if params['trainNoiseLayer']:
				# p_yp_given_y = get_qSC_channel(nb_classes, 0.5)
				p_yp_given_y = torch.eye(nb_classes)
			else:
				p_yp_given_y = get_qSC_channel(nb_classes, params['onp'])
			model, optimizer = getCifar10Model(len(train_loader.dataset), nb_classes,
									appendNoiseLayer = True, p_yp_given_y = p_yp_given_y, 
									noiseTrainable = params['trainNoiseLayer'])
			activationName = 'beforeNoise'
		else:
			model, optimizer = getCifar10Model(len(train_loader.dataset), nb_classes)
			activationName = None

		return model, optimizer, activationName

	# verbose = 0
	if initialModel:
		nb_epoch = 50 // 2
		ctr = 100
		while(ctr > 0):
			model, optimizer, _ = prepareCifarModel()
			for epoch in range(1, nb_epoch + 1):
					# lr_schedule_SGD(optimizer, epoch)
				train_loss = train(model, train_loader, optimizer, epoch, verbose, withNoise = False, lr_schedule=None)
				val_loss, val_acc = validate(model, val_loader)
				if verbose>0:
					print('val loss = %f, val acc = %f'%(val_loss, val_acc))
			if sanityCheckForTrain(model, val_loader, acc_thres = 0.1):
				break
			else:
				print('re-starting model training due to gradients stuck, expiring counter = %d'%(ctr))

			ctr -= 1

	else:
		PATH_curr = initTorchModelPath(root = '../tempTorch', 
									name = 'currTorchModel' + np.str(params['exp_no']) + '.pt')
		saveModel = SaveBestModel(PATH = PATH_curr)

		# ctr = 10 # to take care of gradients not updating, expiring counter
		ctr = 100
		doEarlyStopping = False
		
		if not doEarlyStopping:
			nb_epoch = 50
		

		while(ctr > 0):
			
			model, optimizer, activationName = prepareCifarModel()
			# earlyStop = EarlyStopping(patience = 20 + 80*(ctr<100), PATH = PATH_curr)
			# earlyStop = EarlyStopping(patience = 50 + 50*(ctr<100), PATH = PATH_curr)
			earlyStop = EarlyStopping(patience = 20 + 50*(ctr<100), PATH = PATH_curr)
			# earlyStop = EarlyStopping(patience = 50 + 50, PATH = PATH_curr)
			# earlyStop = EarlyStopping1(patience = 30 + 70*(ctr<100), PATH = PATH_curr)
			schedular = ReduceLROnPlateau(optimizer, 'min', verbose=True, 
												patience = 5, factor = 0.1,
												min_lr = 0.5e-6)
			for epoch in range(1, nb_epoch + 1):
					# lr_schedule_SGD(optimizer, epoch)
				train_loss = train(model, train_loader, optimizer, epoch, verbose, withNoise = trainWithNoise, lr_schedule=None)
				val_loss, val_acc = validate(model, val_loader, activationName=activationName)
				# ex.log_scalar("val_run_loss", val_loss)
				if verbose>0:
					print('val loss = %f, val acc = %f'%(val_loss, val_acc))
				# schedular.step(val_loss)

				if doEarlyStopping:
					toStop = earlyStop.check(model, val_loss)
					# toStop = earlyStop.check(model, val_loss, val_acc)
				else:
					toStop = False
				# saveModel.check(model, val_loss)
				# toStop = False
				if toStop:
					print('early stopping at epoch = %d'%(epoch))
					model, _, _ = prepareCifarModel()
					model.load_state_dict(torch.load(PATH_curr))
					break
			# _, val_acc = validate(model, val_loader)

			# if sanityCheckForTrain(model, val_loader, acc_thres = 0.3):
			if sanityCheckForTrain(model, val_loader, withNoise = trainWithNoise,  acc_thres = 0.2):
				break
			else:
				print('re-starting model training due to gradients stuck, expiring counter = %d'%(ctr))

			ctr -= 1

	return model, train_loss


def trainCifar100Routine(train_loader, val_loader,
						nb_epoch, nb_classes, verbose = 0,
						trainWithNoise=False, params = {}, initialModel = False):

	# schedular = ReduceLROnPlateau(optimizer, 'min', verbose=True, 
			# 								patience = 5, factor = 0.1,
			# 								min_lr = 0.5e-6)
	def prepareCifarModel():

		if trainWithNoise:# adding noise layer
			if params['trainNoiseLayer']:
				# p_yp_given_y = get_qSC_channel(nb_classes, 0.5)
				p_yp_given_y = torch.eye(nb_classes)
			else:
				p_yp_given_y = get_qSC_channel(nb_classes, params['onp'])
			model, optimizer = getCifar100Model(len(train_loader.dataset), nb_classes,
									appendNoiseLayer = True, p_yp_given_y = p_yp_given_y, 
									noiseTrainable = params['trainNoiseLayer'])
			activationName = 'beforeNoise'
		else:
			model, optimizer = getCifar100Model(len(train_loader.dataset), nb_classes)
			activationName = None

		return model, optimizer, activationName

	# verbose = 0
	if initialModel:
		# nb_epoch = 50 // 2
		nb_epoch = 50
		ctr = 100
		while(ctr > 0):
			model, optimizer, _ = prepareCifarModel()
			for epoch in range(1, nb_epoch + 1):
					# lr_schedule_SGD(optimizer, epoch)
				train_loss = train(model, train_loader, optimizer, epoch, verbose, withNoise = False, lr_schedule=None)
				val_loss, val_acc = validate(model, val_loader)
				if verbose>0:
					print('val loss = %f, val acc = %f'%(val_loss, val_acc))
			# if sanityCheckForTrain(model, val_loader, acc_thres = 0.1):
			if sanityCheckForTrain(model, val_loader, acc_thres = 0.05):
				break
			else:
				print('re-starting model training due to gradients stuck, expiring counter = %d'%(ctr))

			ctr -= 1

	else:
		PATH_curr = initTorchModelPath(root = '../tempTorch', 
									name = 'currTorchModel' + np.str(params['exp_no']) + '.pt')
		saveModel = SaveBestModel(PATH = PATH_curr)

		# ctr = 10 # to take care of gradients not updating, expiring counter
		ctr = 100
		doEarlyStopping = False
		
		if not doEarlyStopping:
			nb_epoch = 50
		

		while(ctr > 0):
			
			model, optimizer, activationName = prepareCifarModel()
			# earlyStop = EarlyStopping(patience = 20 + 80*(ctr<100), PATH = PATH_curr)
			# earlyStop = EarlyStopping(patience = 50 + 50*(ctr<100), PATH = PATH_curr)
			earlyStop = EarlyStopping(patience = 20 + 50*(ctr<100), PATH = PATH_curr)
			# earlyStop = EarlyStopping(patience = 50 + 50, PATH = PATH_curr)
			# earlyStop = EarlyStopping1(patience = 30 + 70*(ctr<100), PATH = PATH_curr)
			schedular = ReduceLROnPlateau(optimizer, 'min', verbose=True, 
												patience = 5, factor = 0.1,
												min_lr = 0.5e-6)
			for epoch in range(1, nb_epoch + 1):
					# lr_schedule_SGD(optimizer, epoch)
				train_loss = train(model, train_loader, optimizer, epoch, verbose, withNoise = trainWithNoise, lr_schedule=None)
				val_loss, val_acc = validate(model, val_loader, activationName=activationName)
				# ex.log_scalar("val_run_loss", val_loss)
				if verbose>0:
					print('val loss = %f, val acc = %f'%(val_loss, val_acc))
				# schedular.step(val_loss)

				if doEarlyStopping:
					toStop = earlyStop.check(model, val_loss)
					# toStop = earlyStop.check(model, val_loss, val_acc)
				else:
					toStop = False
				# saveModel.check(model, val_loss)
				# toStop = False
				if toStop:
					print('early stopping at epoch = %d'%(epoch))
					model, _, _ = prepareCifarModel()
					model.load_state_dict(torch.load(PATH_curr))
					break
			# _, val_acc = validate(model, val_loader)

			# if sanityCheckForTrain(model, val_loader, acc_thres = 0.3):
			if sanityCheckForTrain(model, val_loader, withNoise = trainWithNoise,  acc_thres = 0.14):
				break
			else:
				print('re-starting model training due to gradients stuck, expiring counter = %d'%(ctr))

			ctr -= 1

	return model, train_loss


def trainSVHNRoutine_VAAL(train_loader, val_loader, pool_loader,
						nb_epoch, nb_classes, verbose = 0,
						trainWithNoise=False, params = {}, initialModel = False):

	# schedular = ReduceLROnPlateau(optimizer, 'min', verbose=True, 
			# 								patience = 5, factor = 0.1,
			# 								min_lr = 0.5e-6)
	global valid_loss_BASE
	def prepareSVHNModel():

		if trainWithNoise:# adding noise layer
			if params['trainNoiseLayer']:
				# p_yp_given_y = get_qSC_channel(nb_classes, 0.5)
				p_yp_given_y = torch.eye(nb_classes)
			else:
				p_yp_given_y = get_qSC_channel(nb_classes, params['onp'])
			model, optimizer = getSVHNModel_VAAL(len(train_loader.dataset), nb_classes,
									appendNoiseLayer = True, p_yp_given_y = p_yp_given_y, 
									noiseTrainable = params['trainNoiseLayer'])
			activationName = 'beforeNoise'
		else:
			model, optimizer, vae, optim_vae, discriminator, optim_discriminator = getSVHNModel_VAAL(len(train_loader.dataset), nb_classes)
			activationName = None

		return model, optimizer, vae, optim_vae, discriminator, optim_discriminator, activationName

	# verbose = 0
	if initialModel:
		nb_epoch = 50 // 2
		ctr = 100
		while(ctr > 0):
			model, optimizer, _ = prepareSVHNModel()
			for epoch in range(1, nb_epoch + 1):
					# lr_schedule_SGD(optimizer, epoch)
				train_loss = train(model, train_loader, optimizer, epoch, verbose, withNoise=False, lr_schedule=None)
				val_loss, val_acc = validate(model, val_loader)
				if verbose>0:
					print('val loss = %f, val acc = %f'%(val_loss, val_acc))
			if sanityCheckForTrain(model, val_loader, acc_thres = 0.1):
				break
			else:
				print('re-starting model training due to gradients stuck, expiring counter = %d'%(ctr))

			ctr -= 1


	else:
		PATH_curr = initTorchModelPath(root = '../tempTorch', 
									name = 'currTorchModel' + np.str(params['exp_no']) + '.pt')
		saveModel = SaveBestModel(PATH = PATH_curr)

		# ctr = 10 # to take care of gradients not updating, expiring counter
		# ctr = 100
		ctr = 40
		doEarlyStopping = False
		
		if not doEarlyStopping:
			nb_epoch = 100
		

		while(ctr > 0):
			
			model, optimizer, vae, optim_vae, discriminator, optim_discriminator, activationName = prepareSVHNModel()
			# earlyStop = EarlyStopping(patience = 20 + 80*(ctr<100), PATH = PATH_curr)
			earlyStop = EarlyStopping(patience = 50 + 50*(ctr<100), PATH = PATH_curr)
			# earlyStop = EarlyStopping(patience = 20 + 50*(ctr<100), PATH = PATH_curr)
			# earlyStop = EarlyStopping(patience = 50 + 50, PATH = PATH_curr)
			# earlyStop = EarlyStopping1(patience = 30 + 70*(ctr<100), PATH = PATH_curr)
			schedular = ReduceLROnPlateau(optimizer, 'min', verbose=True, 
												patience = 5, factor = np.sqrt(0.1),
												min_lr = 0.5e-6)
			
			epoch = 1
			nb_epoch_MAX = 75
			nb_epoch = nb_epoch_MAX
			# escapedSpuriousTrainingInitMark = False
			escapedEpoch = 0
			while(True):									
				train_loss = train_forVAAL(model, vae, discriminator,
								 train_loader, pool_loader, optimizer,
								 optim_vae, optim_discriminator,
								  epoch, verbose, withNoise = trainWithNoise,
								   lr_schedule=None)
				val_loss, val_acc, ifAllSameOutput = validateForSVHN(model, val_loader, activationName=activationName)
				# ex.log_scalar("val_run_loss", val_loss)
				if verbose>0:
					print('val loss = %f, val acc = %f'%(val_loss, val_acc))
				# schedular.step(val_loss)

				escapedSpuriousTraining = False
				if ifAllSameOutput or val_acc<0.2:
					# start MAX_epoch epochs from this point
					if epoch > 100:
						break
					nb_epoch = nb_epoch_MAX + epoch
					escapedEpoch = 0
					print('All same val outputs, increasing total epochs...')
				else:
					# spurious training escaped
					# nb_epoch = 50
					print('training escaped spurious zone: %d epochs will follow...'%(nb_epoch_MAX))
					escapedSpuriousTraining = True
					if escapedEpoch == 0:
						escapedEpoch = epoch
				epoch += 1

				if epoch > 400:
					break

				if escapedEpoch>0 and (epoch - escapedEpoch > 10) and val_loss > valid_loss_BASE:
					break
			
				# if ifAllSameOutput and epoch>100:
				# 	break

				if doEarlyStopping and not ifAllSameOutput and val_acc>0.2:
					toStop = earlyStop.check(model, val_loss)
					# toStop = earlyStop.check(model, val_loss, val_acc)
				else:
					toStop = False
				
				if not ifAllSameOutput and epoch > nb_epoch:
					break
				# saveModel.check(model, val_loss)
				# toStop = False
				if toStop:
					print('early stopping at epoch = %d'%(epoch))
					model, _, _ = prepareSVHNModel()
					model.load_state_dict(torch.load(PATH_curr))
					break
			# _, val_acc = validate(model, val_loader)

			# if sanityCheckForTrain(model, val_loader, acc_thres = 0.3):
			if sanityCheckForTrainSVHN(model, val_loader, activationName=activationName, acc_thres = 0.2):
				break
			else:
				print('re-starting model training due to gradients stuck, expiring counter = %d'%(ctr))

			ctr -= 1

	return model, train_loss


def trainSVHNRoutine(train_loader, val_loader,
						nb_epoch, nb_classes, verbose = 0,
						trainWithNoise=False, params = {}, initialModel = False):

	# schedular = ReduceLROnPlateau(optimizer, 'min', verbose=True, 
			# 								patience = 5, factor = 0.1,
			# 								min_lr = 0.5e-6)
	global valid_loss_BASE
	def prepareSVHNModel():

		if trainWithNoise:# adding noise layer
			if params['trainNoiseLayer']:
				# p_yp_given_y = get_qSC_channel(nb_classes, 0.5)
				p_yp_given_y = torch.eye(nb_classes)
			else:
				p_yp_given_y = get_qSC_channel(nb_classes, params['onp'])
			model, optimizer = getSVHNModel(len(train_loader.dataset), nb_classes,
									appendNoiseLayer = True, p_yp_given_y = p_yp_given_y, 
									noiseTrainable = params['trainNoiseLayer'])
			activationName = 'beforeNoise'
		else:
			model, optimizer = getSVHNModel(len(train_loader.dataset), nb_classes)
			activationName = None

		return model, optimizer, activationName

	# verbose = 0
	if initialModel:
		nb_epoch = 50 // 2
		ctr = 100
		while(ctr > 0):
			model, optimizer, _ = prepareSVHNModel()
			for epoch in range(1, nb_epoch + 1):
					# lr_schedule_SGD(optimizer, epoch)
				train_loss = train(model, train_loader, optimizer, epoch, verbose, withNoise=False, lr_schedule=None)
				val_loss, val_acc = validate(model, val_loader)
				if verbose>0:
					print('val loss = %f, val acc = %f'%(val_loss, val_acc))
			if sanityCheckForTrain(model, val_loader, acc_thres = 0.1):
				break
			else:
				print('re-starting model training due to gradients stuck, expiring counter = %d'%(ctr))

			ctr -= 1


	else:
		PATH_curr = initTorchModelPath(root = '../tempTorch', 
									name = 'currTorchModel' + np.str(params['exp_no']) + '.pt')
		saveModel = SaveBestModel(PATH = PATH_curr)

		# ctr = 10 # to take care of gradients not updating, expiring counter
		# ctr = 100
		ctr = 40
		doEarlyStopping = False
		
		if not doEarlyStopping:
			nb_epoch = 100
		

		while(ctr > 0):
			
			model, optimizer, activationName = prepareSVHNModel()
			# earlyStop = EarlyStopping(patience = 20 + 80*(ctr<100), PATH = PATH_curr)
			earlyStop = EarlyStopping(patience = 50 + 50*(ctr<100), PATH = PATH_curr)
			# earlyStop = EarlyStopping(patience = 20 + 50*(ctr<100), PATH = PATH_curr)
			# earlyStop = EarlyStopping(patience = 50 + 50, PATH = PATH_curr)
			# earlyStop = EarlyStopping1(patience = 30 + 70*(ctr<100), PATH = PATH_curr)
			schedular = ReduceLROnPlateau(optimizer, 'min', verbose=True, 
												patience = 5, factor = np.sqrt(0.1),
												min_lr = 0.5e-6)
			
			epoch = 1
			nb_epoch_MAX = 75
			nb_epoch = nb_epoch_MAX
			# escapedSpuriousTrainingInitMark = False
			escapedEpoch = 0
			epoch_ctr = 0
			while(True):									
				train_loss = train(model, train_loader, optimizer, epoch, verbose, withNoise = trainWithNoise, lr_schedule=None)
				val_loss, val_acc, ifAllSameOutput = validateForSVHN(model, val_loader, activationName=activationName)
				# ex.log_scalar("val_run_loss", val_loss)
				if verbose>0:
					print('val loss = %f, val acc = %f'%(val_loss, val_acc))
				# schedular.step(val_loss)

				escapedSpuriousTraining = False
				if ifAllSameOutput or val_acc<0.2:
					# start MAX_epoch epochs from this point
					if epoch > 100:
						break
					nb_epoch = nb_epoch_MAX + epoch
					escapedEpoch = 0
					epoch_ctr = 0
					print('All same val outputs, increasing total epochs...')
				else:
					# spurious training escaped
					# nb_epoch = 50
					print('training escaped spurious zone: %d epochs will follow...'%(nb_epoch_MAX - epoch_ctr))
					epoch_ctr += 1
					escapedSpuriousTraining = True
					if escapedEpoch == 0:
						escapedEpoch = epoch
				epoch += 1

				if epoch > 400:
					break

				if escapedEpoch>0 and (epoch - escapedEpoch > 10) and val_loss > valid_loss_BASE:
					break
			
				# if ifAllSameOutput and epoch>100:
				# 	break

				if doEarlyStopping and not ifAllSameOutput and val_acc>0.2:
					toStop = earlyStop.check(model, val_loss)
					# toStop = earlyStop.check(model, val_loss, val_acc)
				else:
					toStop = False
				
				if not ifAllSameOutput and epoch > nb_epoch:
					break
				# saveModel.check(model, val_loss)
				# toStop = False
				if toStop:
					print('early stopping at epoch = %d'%(epoch))
					model, _, _ = prepareSVHNModel()
					model.load_state_dict(torch.load(PATH_curr))
					break
			# _, val_acc = validate(model, val_loader)

			# if sanityCheckForTrain(model, val_loader, acc_thres = 0.3):
			if sanityCheckForTrainSVHN(model, val_loader, activationName=activationName, acc_thres = 0.2):
				break
			else:
				print('re-starting model training due to gradients stuck, expiring counter = %d'%(ctr))

			ctr -= 1

	return model, train_loss


def trainSVHNRoutine_old(train_loader, val_loader,
						nb_epoch, nb_classes, verbose = 0,
						trainWithNoise=False, params = {}, initialModel = False):

	# schedular = ReduceLROnPlateau(optimizer, 'min', verbose=True, 
			# 								patience = 5, factor = 0.1,
			# 								min_lr = 0.5e-6)
	def prepareSVHNModel():

		if trainWithNoise:# adding noise layer
			if params['trainNoiseLayer']:
				# p_yp_given_y = get_qSC_channel(nb_classes, 0.5)
				p_yp_given_y = torch.eye(nb_classes)
			else:
				p_yp_given_y = get_qSC_channel(nb_classes, params['onp'])
			model, optimizer = getSVHNModel(len(train_loader.dataset), nb_classes,
									appendNoiseLayer = True, p_yp_given_y = p_yp_given_y, 
									noiseTrainable = params['trainNoiseLayer'])
			activationName = 'beforeNoise'
		else:
			model, optimizer = getSVHNModel(len(train_loader.dataset), nb_classes)
			activationName = None

		return model, optimizer

	# verbose = 0
	if initialModel:
		nb_epoch = 50 // 2
		ctr = 100
		while(ctr > 0):
			model, optimizer = prepareSVHNModel()
			for epoch in range(1, nb_epoch + 1):
					# lr_schedule_SGD(optimizer, epoch)
				train_loss = train(model, train_loader, optimizer, epoch, verbose, withNoise=False, lr_schedule=None)
				val_loss, val_acc = validate(model, val_loader)
				if verbose>0:
					print('val loss = %f, val acc = %f'%(val_loss, val_acc))
			if sanityCheckForTrain(model, val_loader, acc_thres = 0.1):
				break
			else:
				print('re-starting model training due to gradients stuck, expiring counter = %d'%(ctr))

			ctr -= 1


	else:
		PATH_curr = initTorchModelPath(root = '../tempTorch', 
									name = 'currTorchModel' + np.str(params['exp_no']) + '.pt')
		saveModel = SaveBestModel(PATH = PATH_curr)

		# ctr = 10 # to take care of gradients not updating, expiring counter
		ctr = 100
		doEarlyStopping = True
		
		if not doEarlyStopping:
			nb_epoch = 50
		

		while(ctr > 0):
			
			model, optimizer = prepareSVHNModel()
			# earlyStop = EarlyStopping(patience = 20 + 80*(ctr<100), PATH = PATH_curr)
			earlyStop = EarlyStopping(patience = 50 + 50*(ctr<100), PATH = PATH_curr)
			# earlyStop = EarlyStopping(patience = 20 + 50*(ctr<100), PATH = PATH_curr)
			# earlyStop = EarlyStopping(patience = 50 + 50, PATH = PATH_curr)
			# earlyStop = EarlyStopping1(patience = 30 + 70*(ctr<100), PATH = PATH_curr)
			schedular = ReduceLROnPlateau(optimizer, 'min', verbose=True, 
												patience = 5, factor = np.sqrt(0.1),
												min_lr = 0.5e-6)
			for epoch in range(1, nb_epoch + 1):
					# lr_schedule_SGD(optimizer, epoch)
				train_loss = train(model, train_loader, optimizer, epoch, verbose, withNoise = trainWithNoise, lr_schedule=None)
				val_loss, val_acc = validate(model, val_loader)
				# ex.log_scalar("val_run_loss", val_loss)
				if verbose>0:
					print('val loss = %f, val acc = %f'%(val_loss, val_acc))
				# schedular.step(val_loss)

				if doEarlyStopping:
					toStop = earlyStop.check(model, val_loss)
					# toStop = earlyStop.check(model, val_loss, val_acc)
				else:
					toStop = False
				# saveModel.check(model, val_loss)
				# toStop = False
				if toStop:
					print('early stopping at epoch = %d'%(epoch))
					model, _ = prepareSVHNModel()
					model.load_state_dict(torch.load(PATH_curr))
					break
			# _, val_acc = validate(model, val_loader)

			# if sanityCheckForTrain(model, val_loader, acc_thres = 0.3):
			if sanityCheckForTrain(model, val_loader, withNoise=trainWithNoise, acc_thres = 0.2):
				break
			else:
				print('re-starting model training due to gradients stuck, expiring counter = %d'%(ctr))

			ctr -= 1

	return model, train_loss

	    
def log_performance(scalars, params = {}):

	global log_dir

	writer = SummaryWriter(log_dir = log_dir)

	for key in scalars.keys():
		writer.add_scalar(key, scalars[key].value[-1], scalars[key].index[-1])
	
	writer.close()

class scalarObj(object):
	def __init__(self):
		self.currInd = 0
		self.index = []
		self.value = []

	def add(self, currVal):
		self.value.append(currVal)
		self.index.append(self.currInd)
		self.currInd += 1


dataTypes = ['MNIST', 'rMNIST', 'EMNIST', 'CIFAR10', 'SVHN', 'CIFAR100']
parser = argparse.ArgumentParser(description="AL noisy tests")
parser.add_argument('-data', '--dataType',metavar='Dataset', default='MNIST',
					choices = dataTypes, type=str)
parser.add_argument('-ne', '--Experiments',  metavar='Experiments', default=5,
						type=int)
parser.add_argument('-dri', '--dropout_iterations', default=100,
						help='dropout iterations', type=int)
parser.add_argument('-b', '--batch_size', default=128,
						help='batch size', type=int)
parser.add_argument('-ep', '--nb-epoch', default=50,
					help = 'number of epochs', type=int)
parser.add_argument('-q', '--Queries', default=200,
					help = 'Queries or batch size of AL', type=int)
parser.add_argument('-m', '--method', default='VAAL',
					help = 'Active Learning acquisition method')
parser.add_argument('-n', '--label_noise', default=True, action="store_true",
					help='noisy Oracle')
parser.add_argument('-tn', '--train_noise_layer', default=False, action="store_true",
					help='trian noise layer')
parser.add_argument('-nc', '--noiseChannel', default = 'qSC',
					choices=['qSC', 'noisyTypewriter'], help='type of noisy channel')
parser.add_argument('-np', '--noise-probability', default=0.1,
					help='noise channel probability', type=float)
parser.add_argument('-exp-no', '--experiment_number', default=1,
						help='current experiment number index', type=int)

# @ex.automain

test_acc_obj = scalarObj()
test_loss_obj = scalarObj()
train_loss_obj = scalarObj()
varRatio_obj = scalarObj()
valid_loss_BASE = 0


def main(params):

	# args = parser.parse_args()
	global activation
	global test_loss_obj
	global test_loss_obj
	global train_loss_obj
	global valid_loss_BASE

	for key, value in params.items():
		if key == 'Experiments':
			Experiments = value
		elif key == 'batch_size':
			batch_size = value
		elif key == 'nb_epoch':
			nb_epoch = value
		elif key == 'nb_filters':
			nb_filters = value
		elif key == 'acquisition_iterations':
			acquisition_iterations = value
		elif key == 'dropout_iterations':
			dropout_iterations = value
		elif key == 'Queries':
			Queries = value
		elif key == 'dataType':
			dataType = value
		elif key == 'method':
			method = value
		elif key == 'label_noise':
			label_noise = value
		elif key == 'noiseChannel':
			noiseChannel = value
		elif key == 'onp':
			oracle_noise_probability = value
		elif key == 'mnp':
			model_noise_probability = value

	score=0
	all_accuracy = 0
	Experiments_All_Accuracy = [1]
	Experiments_All_mvRatio = [1]
	PATH_curr = None
	saveDir = os.path.join('Results', dataType, 'New', 'Jatayu', 'NeuRIPS')
	if not os.path.exists(saveDir):
		os.makedirs(saveDir, exist_ok=True)

	for e in range(Experiments):

		print('Experiment Number ', e)

		if dataType == 'MNIST':
			
			nb_classes = 10
			X_train_Base, y_train, X_valid, y_valid, X_test, y_test, X_Pool, y_Pool, nb_classes = getMNISTData(batch_size = batch_size, trainInitSize=params['trainInitSize'])
			acquisition_iterations = int(5)
			poolSizeList = np.ones((acquisition_iterations,),dtype=int) * Queries
			subtract_pixel_mean = False
			lastBatchSkip = False
			shuffleBatches = False
			take_pool_subset = True
			num_pool_subset = 10000
			modelSaveThres = 5
			batchStr = str(Queries)


		elif dataType == 'rMNIST':
			X_train_Base, y_train, X_valid, y_valid, X_test, y_test, X_Pool, y_Pool, nb_classes = get_repeatedMNISTData(batch_size = batch_size, trainInitSize=params['trainInitSize'])
			# acquisition_iterations = int(np.ceil(1000/Queries))
			acquisition_iterations = int(5)
			poolSizeList = np.ones((acquisition_iterations,),dtype=int) * Queries
			subtract_pixel_mean = False
			lastBatchSkip = False
			shuffleBatches = False
			take_pool_subset = True
			modelSaveThres = 5
			batchStr = str(Queries)
			midNumClusters = 2500

			num_pool_subset = 15000


		elif dataType == 'EMNIST':
			X_train_Base, y_train, X_valid, y_valid, X_test, y_test, X_Pool, y_Pool, nb_classes = getEMNISTData(batch_size = batch_size)
			# acquisition_iterations = int(np.ceil(1000/Queries))
			acquisition_iterations = int(5)
			poolSizeList = np.ones((acquisition_iterations,),dtype=int) * Queries
			lastBatchSkip = False
			shuffleBatches = False
			take_pool_subset = True
			modelSaveThres = 5
			batchStr = str(Queries)
			midNumClusters = 2500
			num_pool_subset = 15000

		elif dataType == 'SVHN':
			trainInitSize = 50
			X_train_Base, y_train, X_valid, y_valid, X_test, y_test, X_Pool, y_Pool, nb_classes = getSVHNData(trainInitSize=trainInitSize)
			acquisition_iterations = int(5)
			nb_epoch = 75
			Queries = 5000		
			poolSizeList = np.ones((acquisition_iterations,), dtype=int) * Queries
			batchStr = str(Queries)

			subtract_pixel_mean = False
			lastBatchSkip = False # no need to perform AL on the last batch, as pool is empty
			modelSaveThres = 1

			batch_size = 1024
			shuffleBatches = True
			take_pool_subset = True
			num_pool_subset = 25000
			midNumClusters = 10000

		elif dataType == 'CIFAR10':
			nb_epoch = 50
			trainInitSize = 50
			batch_size = 128
			shuffleBatches= True
			X_train_Base, y_train, X_valid, y_valid, X_test, y_test, X_Pool, y_Pool, nb_classes = getCIFAR10Data(trainInitSize=trainInitSize)
			Queries = 5000
			acquisition_iterations = int(6)
			poolSizeList = np.ones((acquisition_iterations,), dtype=int) * Queries
			batchStr = str(Queries)

			subtract_pixel_mean = False
			# lastBatchSkip = True # no need to perform AL on the last batch, as pool is empty
			lastBatchSkip = False
			modelSaveThres = 1
			take_pool_subset = True
			num_pool_subset = 25000

		elif dataType == 'CIFAR100':
			X_train_Base, y_train, X_valid, y_valid, X_test, y_test, X_Pool, y_Pool, nb_classes = getCIFAR100Data()
			shuffleBatches = True
			nb_epoch = 50
			batch_size = 128
			Queries = 8000
			acquisition_iterations = int(4)
			poolSizeList = np.ones((acquisition_iterations,), dtype=int) * Queries
			batchStr = str(Queries)

			subtract_pixel_mean = False
			# lastBatchSkip = True # no need to perform AL on the last batch, as pool is empty
			lastBatchSkip = False
			modelSaveThres = 1
			take_pool_subset = True
			num_pool_subset = 25000

		print(poolSizeList)

		X_train = X_train_Base.clone()

		print('X_train shape:', X_train.shape)
		print(X_train.shape[0], 'train samples')

		print('Distribution of Training Classes:', np.bincount(y_train.numpy()))

		print('Training Model Without Acquisitions in Experiment', e)

		train_use = data_utils.TensorDataset(X_train, y_train)
		train_loader = data_utils.DataLoader(train_use, batch_size=batch_size, shuffle=shuffleBatches)

		val_use = data_utils.TensorDataset(X_valid, y_valid)
		val_loader = data_utils.DataLoader(val_use, batch_size=len(val_use), shuffle=False)

		test_use = data_utils.TensorDataset(X_test, y_test)
		test_loader = data_utils.DataLoader(test_use, batch_size=1024, shuffle=False)

		pool_subset = torch.from_numpy(np.arange(X_Pool.shape[0]))
		if take_pool_subset:
			if num_pool_subset < X_Pool.shape[0]:
				pool_subset = torch.from_numpy(np.random.choice(X_Pool.shape[0], num_pool_subset))
    	
		pool_d = data_utils.TensorDataset(X_Pool, y_Pool)
		unlabeled_sampler = data_utils.sampler.SubsetRandomSampler(pool_subset)
		pool_loader = data_utils.DataLoader(pool_d, batch_size=batch_size, shuffle=False,
												sampler=unlabeled_sampler)


		# classWeight = torch.from_numpy(np.bincount(y_train.numpy())).float()
		if dataType == 'MNIST' or dataType == 'rMNIST':
			model, optimizer, vae, optim_vae, discriminator, optim_discriminator = getModelForMNIST_VAAL(X_train.shape[0], nb_classes)
			for epoch in range(1, nb_epoch + 1):
				train_loss = train_forVAAL(model, vae, discriminator, train_loader,
									 pool_loader, optimizer, optim_vae,
									 optim_discriminator, epoch, verbose=0, withNoise = False)

		elif dataType == 'EMNIST':
			train_loss = 0
			model, optimizer, vae, optim_vae, discriminator, optim_discriminator = getModelForEMNIST_VAAL(X_train.shape[0], nb_classes)
			for epoch in range(1, nb_epoch + 1):
				train_loss = train_forVAAL(model, vae, discriminator, train_loader,
									 pool_loader, optimizer, optim_vae,
									 optim_discriminator, epoch, verbose=0, withNoise = False)
				# train_loss, train_acc = validate(model, train_loader, optimizer, epoch)			

		elif dataType == 'CIFAR10':
			model, optimizer, vae, optim_vae, discriminator, optim_discriminator = getCifar10Model_VAAL(len(train_loader.dataset), nb_classes)		
			nb_epoch = 50	
			for epoch in range(1, nb_epoch//2+1):
				train_loss = train_forVAAL(model, vae, discriminator, train_loader,
										pool_loader, optimizer, optim_vae,
										optim_discriminator, epoch, verbose=1, withNoise = False)
				val_loss, val_acc = validate(model, val_loader, activationName=None)
					# if verbose>0:
				print('val loss = %f, val acc = %f'%(val_loss, val_acc))

		elif dataType == 'SVHN':
			model, optimizer, vae, optim_vae, discriminator, optim_discriminator = getSVHNModel_VAAL(len(train_loader.dataset), nb_classes)		
			nb_epoch = 50	
			for epoch in range(1, nb_epoch//2+1):
				train_loss = train_forVAAL(model, vae, discriminator, train_loader,
										pool_loader, optimizer, optim_vae,
										optim_discriminator, epoch, verbose=1, withNoise = False)
				val_loss, val_acc = validate(model, val_loader, activationName=None)
					# if verbose>0:
				print('val loss = %f, val acc = %f'%(val_loss, val_acc))


		elif dataType == 'CIFAR100':
			model, optimizer, vae, optim_vae, discriminator, optim_discriminator = getCifar100Model_VAAL(len(train_loader.dataset), nb_classes)		
			nb_epoch = 50	
			for epoch in range(1, nb_epoch//2+1):
				train_loss = train_forVAAL(model, vae, discriminator, train_loader,
										pool_loader, optimizer, optim_vae,
										optim_discriminator, epoch, verbose=1, withNoise = False)
				val_loss, val_acc = validate(model, val_loader, activationName=None)
					# if verbose>0:
				print('val loss = %f, val acc = %f'%(val_loss, val_acc))

		test_loss, test_acc = test(model, test_loader)

		valid_loss_BASE, _ = validate(model, val_loader)

		test_loss_obj.add(test_loss)
		test_acc_obj.add(test_acc)
		train_loss_obj.add(train_loss)
		# test_loss_obj.add(test_loss)

		# log_performance({'test_loss':test_loss_obj, 'test_acc':test_acc_obj,
		# 				'train_loss':train_loss_obj}, params)

		all_accuracy = np.array(test_acc)
		all_mvRatio = np.array([])
		mvRatio = 0

		print('Starting Active Learning in Experiment ', e)

		firstCall = True
		trainWithNoise = False

		for i in range(acquisition_iterations):

			if i < 1:
				doRandom = False
			else:
				doRandom = False
	
			start_time = time()
			print('POOLING ITERATION %d with method=%s' %(i, method), end='')
			if (i == acquisition_iterations - 1) & (lastBatchSkip is True):
				x_pool_index = torch.from_numpy( np.arange(X_Pool.shape[0]))
			else:

				if method == 'VAAL' or method == 'VAAL-noise':
					extraParams = {'take_pool_subset':take_pool_subset}
					extraParams['num_pool_subset'] = num_pool_subset
					extraParams['trainWithNoise'] = trainWithNoise
					x_pool_index = getVAALIndices(pool_loader, vae, discriminator, 
													poolSizeList[i], nb_classes, batch_size, 
													extraParams)
				else:
					print('UNKNOWN METHOD, will quit...')
					return 1

			print(', time taken=%f'%(time()-start_time))
			strOut = 'AL_' + method + '_b_' + batchStr
			if method.split('-')[-1] == 'noise':trainWithNoise=True

			Pooled_X = X_Pool[pool_subset[x_pool_index], :,:,:].numpy()
			Pooled_Y = y_Pool[pool_subset[x_pool_index]].numpy()

			if label_noise is True:
				if noiseChannel == 'noisyTypewriter':
					Pooled_Y = getNoisyTypeWriter_channelOut(Pooled_Y, nb_classes, oracle_noise_probability)
				elif noiseChannel == 'qSC':
					Pooled_Y_Orig = np.copy(Pooled_Y)
					Pooled_Y = get_qSC_channelOut(Pooled_Y, nb_classes, oracle_noise_probability)
					print('oracle differs in', str(np.sum(Pooled_Y_Orig != Pooled_Y)), ' places')
				
			x_pool_index_orig = pool_subset[x_pool_index]
			X_train, y_train, X_Pool, y_Pool = updateTrainAndPoolPoints(X_train, y_train, 
												X_Pool, y_Pool, Pooled_X, Pooled_Y, x_pool_index_orig)		

			print('Acquised Points added to training set, new training size = %d, new pool size = %d'%(X_train.shape[0], X_Pool.shape[0]))

			print('Distribution of Training Classes:', np.bincount(y_train))
			
			train_use = data_utils.TensorDataset(X_train, y_train)
			train_loader = data_utils.DataLoader(train_use, batch_size=batch_size, shuffle=shuffleBatches)

			pool_subset = torch.from_numpy(np.arange(X_Pool.shape[0]))
			if take_pool_subset:
				if num_pool_subset < X_Pool.shape[0]:
					pool_subset = torch.from_numpy(np.random.choice(X_Pool.shape[0], num_pool_subset))
			

			pool_d = data_utils.TensorDataset(X_Pool, y_Pool)
			unlabeled_sampler = data_utils.sampler.SubsetRandomSampler(pool_subset)
			pool_loader = data_utils.DataLoader(pool_d, batch_size=batch_size, shuffle=False,
												sampler=unlabeled_sampler)

			# classWeight = torch.from_numpy( np.bincount(y_train.numpy())).float()
			if dataType == 'MNIST' or dataType == 'rMNIST':
				if Queries < 200: # save best model in case data samples are very less
					params['saveModel'] = True
				else:
					params['saveModel'] = False
				model, train_loss = trainMNIST_VAALRoutine(train_loader, val_loader, pool_loader,
											nb_epoch, nb_classes, verbose=0,
											trainWithNoise=trainWithNoise, params = params)
				
			elif dataType == 'EMNIST':
				if Queries < 200: # save best model in case data samples are very less
					params['saveModel'] = True
				else:
					params['saveModel'] = False
				model, train_loss = trainEMNIST_VAALRoutine(train_loader, val_loader, pool_loader,
									nb_epoch, nb_classes, verbose=0,
									trainWithNoise=trainWithNoise, params = params)

			elif dataType == 'CIFAR10':
				start_time = time()
				model, optimizer, vae, optim_vae, discriminator, optim_discriminator = getCifar10Model_VAAL(len(train_loader.dataset), nb_classes)			
				for epoch in range(1, nb_epoch+1):
					train_loss = train_forVAAL(model, vae, discriminator, train_loader,
											pool_loader, optimizer, optim_vae,
											optim_discriminator, epoch, verbose=1, withNoise = False)
					val_loss, val_acc = validate(model, val_loader, activationName=None)
					# if verbose>0:
					print('val loss = %f, val acc = %f'%(val_loss, val_acc))

			elif dataType == 'SVHN':
				start_time = time()
				model, train_loss = trainSVHNRoutine_VAAL(train_loader, val_loader, pool_loader,
						nb_epoch, nb_classes, verbose = 1,
						trainWithNoise=trainWithNoise, params = params)
			
			elif dataType == 'CIFAR100':
				start_time = time()
				model, optimizer, vae, optim_vae, discriminator, optim_discriminator = getCifar100Model_VAAL(len(train_loader.dataset), nb_classes)			
				for epoch in range(1, nb_epoch+1):
					train_loss = train_forVAAL(model, vae, discriminator, train_loader,
											pool_loader, optimizer, optim_vae,
											optim_discriminator, epoch, verbose=1, withNoise = False)
					val_loss, val_acc = validate(model, val_loader, activationName=None)
					# if verbose>0:
					print('val loss = %f, val acc = %f'%(val_loss, val_acc))

			print('Evaluate Model Test Accuracy with pooled points')

			if trainWithNoise:
				test_loss, test_acc = test(model, test_loader, activationName = 'beforeNoise')
			else:
				test_loss, test_acc = test(model, test_loader)

			test_loss_obj.add(test_loss)
			test_acc_obj.add(test_acc)
			train_loss_obj.add(train_loss)
			# test_loss_obj.add(test_loss)

			# log_performance({'test_loss':test_loss_obj, 'test_acc':test_acc_obj,
			# 				'train_loss':train_loss_obj}, params)
			
			print('Test loss:', test_loss)
			print('Test accuracy:', test_acc)
			all_accuracy = np.append(all_accuracy, test_acc)
			all_mvRatio = np.append(all_mvRatio, mvRatio)

			if trainWithNoise:
				print(model.nl.linear.weight)

			if i % modelSaveThres == 0:
				Experiments_All_Accuracy[e] = all_accuracy
				Experiments_All_mvRatio[e] = all_mvRatio
				outData = {'Experiments_All_Accuracy': Experiments_All_Accuracy,
							'Experiments_All_mvRatio':Experiments_All_mvRatio}
				fileNameOut = strOut + '_' + dataType + '_Experiment_'+np.str(params['exp_no'])+'.p'
				pk.dump(outData, open(os.path.join(saveDir, fileNameOut), 'wb'))


		print('Storing Accuracy Values over experiments')
		Experiments_All_Accuracy[e] = all_accuracy
		Experiments_All_mvRatio[e] = all_mvRatio          
		outData = {'Experiments_All_Accuracy': Experiments_All_Accuracy,
						'Experiments_All_mvRatio':Experiments_All_mvRatio}
		fileNameOut = strOut + '_' + dataType + '_Experiment_'+np.str(params['exp_no'])+'.p'
		pk.dump(outData, open(os.path.join(saveDir, fileNameOut), 'wb'))
		Experiments_All_Accuracy = Experiments_All_Accuracy + [1]
		Experiments_All_mvRatio = Experiments_All_mvRatio + [1]

	Average_Accuracy = np.divide(Experiments_All_Accuracy, Experiments)

	if PATH_curr is not None:
		print('removing saved torch temporary model...')
		strUse = 'rm ' + PATH_curr
		os.system(strUse)

	return 1

if __name__ == '__main__':

	global log_dir

	args = parser.parse_args()
	params = cfg()

	params['dataType'] = args.dataType
	params['method'] = args.method
	params['exp_no'] = args.experiment_number
	params['trainNoiseLayer'] = args.train_noise_layer
	params['onp'] = args.noise_probability
	params['label_noise'] = args.label_noise
	params['Experiments'] = args.Experiments
	params['Queries'] = args.Queries

	strName = params['method'] + '_onp_' + np.str(params['onp']) + '_b_' + np.str(args.Queries) + '_Exp_' + np.str(params['exp_no'])
	log_dir = os.path.join('../tempTorch',
                           'tensorflow', params['dataType'],
						   'logs', strName)

	main(params)
