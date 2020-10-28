import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProbabilitySimplex(object):
    def __init__(self):
        pass
        
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data.clamp_(0)
            w.div_(torch.tensor(1e-8) + torch.sum(w, 0, keepdim=True).expand_as(w))
            module.weight.data = w


def get_qSC_channel(nb_classes, epsilon = 0.5):

    prob_yp_given_y = np.ones((nb_classes, nb_classes)) * (epsilon / (nb_classes-1))
    np.fill_diagonal(prob_yp_given_y,1-epsilon)
    
    return torch.from_numpy(prob_yp_given_y).float()


class NoiseLayer(nn.Module):
	def __init__(self, nb_classes, p_yp_given_y = None, trainable = False):
		super(NoiseLayer, self).__init__()
		
		self.linear = nn.Linear(nb_classes, nb_classes, bias=False)
		
		if p_yp_given_y is None:
			p_yp_given_y = get_qSC_channel(nb_classes, 0.5)
			self.linear.weight.data = p_yp_given_y
		else:
			self.linear.weight.data = p_yp_given_y
		
		if not trainable:
			self.linear.weight.requires_grad = False
		
	def forward(self, x):
		
		x = self.linear(x)
		return x


class MNIST_NN(nn.Module):
	def __init__(self):
		super(MNIST_NN, self).__init__()

		self.convBlock = nn.Sequential(
			nn.Conv2d(1, 32, 4, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, 4, 1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.25)
		)
		self.Dense = nn.Linear(32 * 11 * 11, 128)
		self.linearBlock = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(128, 10)
		)

	def forward(self, x):

		x = self.convBlock(x)
		x = x.view(-1, 32 * 11 * 11)
		x_d = self.Dense(x)
		x = self.linearBlock(x_d)
		return x, F.relu(x_d)


class MNIST_NN_withNoise(nn.Module):
	def __init__(self, p_yp_given_y = None, noiseTrainable = False):
		super(MNIST_NN_withNoise, self).__init__()

		self.convBlock = nn.Sequential(
			nn.Conv2d(1, 32, 4, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, 4, 1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.25)
		)
		self.Dense = nn.Linear(32 * 11 * 11, 128)
		self.linearBlock = nn.Sequential(
			nn.ReLU(),
			nn.Dropout2d(0.5),
			nn.Linear(128, 10)
		)
		self.nl = NoiseLayer(10, p_yp_given_y = p_yp_given_y, trainable = noiseTrainable)

	def forward(self, x):

		x = self.convBlock(x)
		x = x.view(-1, 32 * 11 * 11)
		x_d = self.Dense(x)
		x_lb = self.linearBlock(x_d)

		x = F.softmax(x_lb, dim=1)
		# x = F.log_softmax(x_lb, dim=1)
		# x = torch.exp(x)
		x = self.nl(x)
		x = torch.log(x)  # return log of probabilities to mimic log_softmax, after this we compute nll loss

		return x, x_lb, F.relu(x_d)


def getModelForMNIST(numTrain, nb_classes, appendNoiseLayer = False,
						 p_yp_given_y = None, noiseTrainable = False):

	# make sure to add 'output' as name for the output layer, 
	# and add name 'kcd' for which coreset
	# output is needed as the delta function, and 'kcd-and-output' for which output 
	# and coreset delta is same

	if appendNoiseLayer:
		model = MNIST_NN_withNoise(p_yp_given_y = p_yp_given_y, noiseTrainable = noiseTrainable)
	else:
		model = MNIST_NN()
		# model = MNIST_NN1()
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
	
	return model, optimizer

class EMNIST_NN(nn.Module):
	def __init__(self):
		super(EMNIST_NN, self).__init__()

		self.convBlock = nn.Sequential(
			nn.Conv2d(1, 32, 4, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, 4, 1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.25),

			nn.Conv2d(64, 128, 4, 1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.25)
		)
		self.Dense = nn.Linear(128 * 4 * 4, 512)
		self.linearBlock = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(512, 47)
		)

	def forward(self, x):

		x = self.convBlock(x)
		x = x.view(-1, 128 * 4 * 4)
		x_d = self.Dense(x)
		x = self.linearBlock(x_d)
		return x, F.relu(x_d)


class EMNIST_NN_withNoise(nn.Module):
	def __init__(self, p_yp_given_y = None, noiseTrainable = False):
		super(EMNIST_NN_withNoise, self).__init__()

		self.convBlock = nn.Sequential(
			nn.Conv2d(1, 32, 4, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, 4, 1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.25),

			nn.Conv2d(64, 128, 4, 1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.25)
		)
		self.Dense = nn.Linear(128 * 4 * 4, 512)
		self.linearBlock = nn.Sequential(
			nn.ReLU(),
			nn.Dropout2d(0.5),
			nn.Linear(512, 47)
		)
		self.nl = NoiseLayer(47, p_yp_given_y = p_yp_given_y, trainable = noiseTrainable)

	def forward(self, x):

		x = self.convBlock(x)
		x = x.view(-1, 128 * 4 * 4)
		x_d = self.Dense(x)
		x_lb = self.linearBlock(x_d)

		x = F.softmax(x_lb, dim=1)
		x = self.nl(x)
		x = torch.log(x)  # return log of probabilities to mimic log_softmax, after this we compute nll loss

		return x, x_lb, F.relu(x_d)


def getModelForEMNIST(numTrain, nb_classes, appendNoiseLayer = False,
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
			{'params':model.nl.parameters(), 'weight_decay':0.5}
			]
	else:
		paramsForOptim = [
			{'params':model.convBlock.parameters()},
			{'params':model.linearBlock.parameters()},
			{'params':model.Dense.parameters(), 'weight_decay': decay}
			]
	optimizer = torch.optim.Adam(paramsForOptim, lr =lr)
	
	return model, optimizer


class VGG16_use(nn.Module):
	def __init__(self):
		super(VGG16_use, self).__init__()

		self.convBlock = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, padding=1),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Dropout2d(0.25),

			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Dropout2d(0.25),

			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Dropout2d(0.25),

			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Dropout2d(0.25),

			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			# nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Dropout2d(0.25),
		)
		self.Dense1 = nn.Linear(512*4, 512)
		self.dropOut1 = nn.Dropout(0.5)
		self.Dense2 = nn.Linear(512, 512)

		self.linearBlock = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(512, 10)
		)

	def forward(self, x):
		x = self.convBlock(x)
		x = x.view(-1, 512*4)
		x = F.relu(self.Dense1(x))
		x = self.dropOut1(x)
		x_d = F.relu(self.Dense2(x))
		x_l = self.linearBlock(x_d)

		return x_l, F.relu(x_d)


class CIFAR_NN_SimpleNext(nn.Module):
	def __init__(self):
		super(CIFAR_NN_SimpleNext, self).__init__()

		self.convBlock = nn.Sequential(
			nn.Conv2d(3, 32, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.25),

			nn.Conv2d(32, 64, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.25),

			nn.Conv2d(64, 128, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.25)
		)
		self.Dense = nn.Linear(128 * 4 * 4, 512//4)
		self.linearBlock = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(512//4, 10)
		)

	def forward(self, x):

		x = self.convBlock(x)
		x = x.view(-1, 128 * 4 * 4)
		# x = x.view(x.size(0), -1)
		x_d = self.Dense(x)
		x = self.linearBlock(x_d)
		return x, F.relu(x_d)


class CIFAR_NN_SimpleNext_withNoise(nn.Module):
	def __init__(self, p_yp_given_y = None, noiseTrainable = False):
		super(CIFAR_NN_SimpleNext_withNoise, self).__init__()

		self.convBlock = nn.Sequential(
			nn.Conv2d(3, 32, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.25),

			nn.Conv2d(32, 64, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.25),

			nn.Conv2d(64, 128, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.25)
		)
		self.Dense = nn.Linear(128 * 4 * 4, 512)
		self.linearBlock = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(512, 10)
		)
		self.nl = NoiseLayer(10, p_yp_given_y = p_yp_given_y, 
								trainable = noiseTrainable)

	def forward(self, x):

		x = self.convBlock(x)
		x = x.view(-1, 128 * 4 * 4)
		# x = x.view(x.size(0), -1)
		x_d = self.Dense(x)
		x_lb = self.linearBlock(x_d)

		x = F.softmax(x_lb, dim=1)
		# x = F.log_softmax(x_lb, dim=1)
		# x = torch.exp(x)

		x = self.nl(x)
		x = torch.log(x)  # return log of probabilities to mimic log_softmax, after this we compute nll loss
		return x, x_lb, F.relu(x_d)


class CIFAR100_NN_SimpleNext(nn.Module):
	def __init__(self):
		super(CIFAR100_NN_SimpleNext, self).__init__()

		self.convBlock = nn.Sequential(
			nn.Conv2d(3, 32, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.25),

			nn.Conv2d(32, 64, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.25),

			nn.Conv2d(64, 128, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.25)
		)
		self.Dense = nn.Linear(128 * 4 * 4, 512)
		self.linearBlock = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(512, 100)
		)

	def forward(self, x):

		x = self.convBlock(x)
		x = x.view(-1, 128 * 4 * 4)
		# x = x.view(x.size(0), -1)
		x_d = self.Dense(x)
		x = self.linearBlock(x_d)
		return x, F.relu(x_d)


class CIFAR100_NN_SimpleNext_withNoise(nn.Module):
	def __init__(self, p_yp_given_y = None, noiseTrainable = False):
		super(CIFAR100_NN_SimpleNext_withNoise, self).__init__()

		self.convBlock = nn.Sequential(
			nn.Conv2d(3, 32, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.25),

			nn.Conv2d(32, 64, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.25),

			nn.Conv2d(64, 128, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.25)
		)
		self.Dense = nn.Linear(128 * 4 * 4, 512)
		self.linearBlock = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(512, 100)
		)
		self.nl = NoiseLayer(100, p_yp_given_y = p_yp_given_y, 
								trainable = noiseTrainable)

	def forward(self, x):

		x = self.convBlock(x)
		x = x.view(-1, 128 * 4 * 4)
		# x = x.view(x.size(0), -1)
		x_d = self.Dense(x)
		x_lb = self.linearBlock(x_d)

		x = F.softmax(x_lb, dim=1)
		# x = F.log_softmax(x_lb, dim=1)
		# x = torch.exp(x)

		x = self.nl(x)
		x = torch.log(x)  # return log of probabilities to mimic log_softmax, after this we compute nll loss
		return x, x_lb, F.relu(x_d)


class SVHN_NN_SimpleNext(nn.Module):
	def __init__(self):
		super(SVHN_NN_SimpleNext, self).__init__()

		self.convBlock = nn.Sequential(
			nn.Conv2d(3, 32, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.3),

			nn.Conv2d(32, 64, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.3),

			nn.Conv2d(64, 128, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.3)
		)
		self.Dense = nn.Linear(128 * 4 * 4, 512//4)
		self.linearBlock = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(512//4, 10)
		)

	def forward(self, x):

		x = self.convBlock(x)
		x = x.view(-1, 128 * 4 * 4)
		# x = x.view(x.size(0), -1)
		x_d = self.Dense(x)
		x = self.linearBlock(x_d)
		return x, F.relu(x_d)


class SVHN_NN_SimpleNext_withNoise(nn.Module):
	def __init__(self, p_yp_given_y = None, noiseTrainable = False):
		super(SVHN_NN_SimpleNext_withNoise, self).__init__()

		self.convBlock = nn.Sequential(
			nn.Conv2d(3, 32, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.25),

			nn.Conv2d(32, 64, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.25),

			nn.Conv2d(64, 128, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, 3, 1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.25)
		)
		self.Dense = nn.Linear(128 * 4 * 4, 512)
		self.linearBlock = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(512, 10)
		)
		self.nl = NoiseLayer(10, p_yp_given_y = p_yp_given_y, 
								trainable = noiseTrainable)

	def forward(self, x):

		x = self.convBlock(x)
		x = x.view(-1, 128 * 4 * 4)
		# x = x.view(x.size(0), -1)
		x_d = self.Dense(x)
		x_lb = self.linearBlock(x_d)

		x = F.softmax(x_lb, dim=1)
		# x = F.log_softmax(x_lb, dim=1)
		# x = torch.exp(x)

		x = self.nl(x)
		x = torch.log(x)  # return log of probabilities to mimic log_softmax, after this we compute nll loss
		return x, x_lb, F.relu(x_d)


def getCifar10Model(numTrain, nb_classes, appendNoiseLayer = False,
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
	return model, optimizer


def getSVHNModel(numTrain, nb_classes, appendNoiseLayer = False,
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
		# model = VGG16_use()
		# optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
	model.to(device)

	# summary(model, (3, 32, 32))

	
	# if appendNoiseLayer:
	# else:
	return model, optimizer


def getCifar100Model(numTrain, nb_classes, appendNoiseLayer = False,
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

	
	# if appendNoiseLayer:
	# else:
	return model, optimizer