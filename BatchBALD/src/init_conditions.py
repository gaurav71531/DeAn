import torch
from torchvision import datasets, transforms
import pickle as pk
import numpy as np

from models import *


def getMNISTInitConditions(e = 0, returnType = 'data'):

	if returnType == 'data':

		dataMain = pk.load(open('../tempTorch/MNIST_Experiment_Init_Data_evenSplit_' + str(e) + '.p', 'rb'))
		idx_train = dataMain['idx_train']

		kwargs = {'num_workers':1, 'pin_memory': True}

		batch_size = 128
		train_loader = torch.utils.data.DataLoader(
				datasets.MNIST('../data', train=True, download=True,
							transform=transforms.ToTensor()
								),
				batch_size=batch_size, shuffle=False, **kwargs)

		test_dataset = datasets.MNIST('../data', train=False, transform=transforms.ToTensor())
		test_loader = torch.utils.data.DataLoader(test_dataset,
				batch_size=len(test_dataset), shuffle=False, **kwargs)

		train_data_all = train_loader.dataset.data
		train_target_all = train_loader.dataset.targets

		test_data = test_loader.dataset.data
		test_target = test_loader.dataset.targets

		train_data_val = train_data_all[10000:15000, :,:]
		train_target_val = train_target_all[10000:15000]
		train_data_pool = train_data_all[20000:60000, :,:]
		train_target_pool = train_target_all[20000:60000]

		train_data_all = train_data_all[:10000,:,:]
		train_target_all = train_target_all[:10000]

		train_data = train_data_all[idx_train,:,:]
		train_target = train_target_all[idx_train]

		train_data_val.unsqueeze_(1)
		train_data_pool.unsqueeze_(1)
		train_data.unsqueeze_(1)
		test_data.unsqueeze_(1)

		train_data_pool = train_data_pool.float()
		train_data_val = train_data_val.float()
		train_data = train_data.float()
		test_data = test_data.float()

		train_data_val /= 255
		train_data_pool /= 255
		train_data /= 255

		return (train_data, train_target, 
				train_data_val, train_target_val,
				test_data/255, test_target,
				train_data_pool, train_target_pool,
				dataMain['batch_size'], dataMain['shuffleBatches'],
				dataMain['nb_epoch']
				)

	elif returnType == 'model':
		model, _ = getModelForMNIST(40, 10)
		PATH_curr = '../tempTorch/MNIST_Init_model_Experiment_evenSplit_' + np.str(e) + '.pt'
		model.load_state_dict(torch.load(PATH_curr))
		return model


def getMNISTData(batch_size = 128, trainInitSize = 20):

# the data, shuffled and split between tran and test sets


	kwargs = {'num_workers':1, 'pin_memory': True}

	train_loader = torch.utils.data.DataLoader(
			datasets.MNIST('../data', train=True, download=True,
						transform=transforms.ToTensor()
							),
			batch_size=batch_size, shuffle=False, **kwargs)

	test_dataset = datasets.MNIST('../data', train=False, transform=transforms.ToTensor())
	test_loader = torch.utils.data.DataLoader(test_dataset,
			batch_size=len(test_dataset), shuffle=False, **kwargs)

	train_data_all = train_loader.dataset.data
	train_target_all = train_loader.dataset.targets

	test_data = test_loader.dataset.data
	test_target = test_loader.dataset.targets

	# train_data = train_data_all.numpy()
	# train_target = train_target_all.numpy()

	train_data_val = train_data_all[10000:15000, :,:]
	train_target_val = train_target_all[10000:15000]
	train_data_pool = train_data_all[20000:60000, :,:]
	train_target_pool = train_target_all[20000:60000]

	train_data_val.unsqueeze_(1)
	train_data_pool.unsqueeze_(1)
	train_data_all.unsqueeze_(1)
	test_data.unsqueeze_(1)

	train_data_pool = train_data_pool.float()
	train_data_val = train_data_val.float()
	train_data_all = train_data_all.float()
	test_data = test_data.float()

	train_data = []
	train_target = []
	numInit = np.floor(trainInitSize/10)*10
	numEachClass = int(numInit/10)

	for i in range(0,10):
		arr = np.array(np.where(train_target_all.numpy()==i))
		idx = np.random.permutation(arr)
		data_i =  train_data_all.numpy()[ idx[0][0:numEachClass], :,:,: ]
		target_i = train_target_all.numpy()[idx[0][0:numEachClass]]
		train_data.append(data_i)
		train_target.append(target_i)

	train_data = np.concatenate(train_data, axis = 0).astype("float32")
	train_target = np.concatenate(train_target, axis=0)

	# train_data = train_data_all.numpy()
	# train_target = train_target_all.numpy()

	# train_data, train_target = torch.from_numpy(train_data/255).float(), torch.from_numpy(train_target)

	return torch.from_numpy(train_data/255).float(), torch.from_numpy(train_target), train_data_val/255,train_target_val, test_data/255, test_target, train_data_pool/255, train_target_pool, 10


def get_repeatedMNISTData(batch_size = 128, trainInitSize = 20):

# the data, shuffled and split between tran and test sets
	num_repetations = 3 # applying repetations to the pool data
	noise_var = 0.1


	kwargs = {'num_workers':1, 'pin_memory': True}

	train_loader = torch.utils.data.DataLoader(
			datasets.MNIST('../data', train=True, download=True,
						transform=transforms.ToTensor()
							),
			batch_size=batch_size, shuffle=False, **kwargs)

	test_dataset = datasets.MNIST('../data', train=False, transform=transforms.ToTensor())
	test_loader = torch.utils.data.DataLoader(test_dataset,
			batch_size=len(test_dataset), shuffle=False, **kwargs)

	train_data_all = train_loader.dataset.data
	train_target_all = train_loader.dataset.targets

	test_data = test_loader.dataset.data
	test_target = test_loader.dataset.targets

	# train_data = train_data_all.numpy()
	# train_target = train_target_all.numpy()

	train_data_val = train_data_all[10000:15000, :,:]
	train_target_val = train_target_all[10000:15000]

	train_data_pool = train_data_all[20000:60000, :,:]
	train_target_pool = train_target_all[20000:60000]
	train_data_pool = np.repeat(train_data_pool, num_repetations, axis=0)
	train_target_pool = np.repeat(train_target_pool, num_repetations, axis=0)

	train_data_pool /= 255
	train_data_pool += np.random.normal(0, noise_var, np.shape(train_target_pool))

	train_data_val.unsqueeze_(1)
	train_data_pool.unsqueeze_(1)
	train_data_all.unsqueeze_(1)
	test_data.unsqueeze_(1)

	train_data_pool = train_data_pool.float()
	train_data_val = train_data_val.float()
	train_data_all = train_data_all.float()
	test_data = test_data.float()

	train_data = []
	train_target = []
	numInit = np.floor(trainInitSize/10)*10
	numEachClass = int(numInit/10)

	for i in range(0,10):
		arr = np.array(np.where(train_target_all.numpy()==i))
		idx = np.random.permutation(arr)
		data_i =  train_data_all.numpy()[ idx[0][0:numEachClass], :,:,: ]
		target_i = train_target_all.numpy()[idx[0][0:numEachClass]]
		train_data.append(data_i)
		train_target.append(target_i)

	train_data = np.concatenate(train_data, axis = 0).astype("float32")
	train_target = np.concatenate(train_target, axis=0)

	# train_data = train_data_all.numpy()
	# train_target = train_target_all.numpy()

	# train_data, train_target = torch.from_numpy(train_data/255).float(), torch.from_numpy(train_target)

	return torch.from_numpy(train_data/255).float(), torch.from_numpy(train_target), train_data_val/255,train_target_val, test_data/255, test_target, train_data_pool, train_target_pool, 10


def getSVHNInitConditions(e = 0, returnType = 'data'):

	if returnType == 'data':
		dataMain = pk.load(open('../tempTorch/SVHN_Experiment_Init_Data_evenSplit_' + str(e) + '.p', 'rb'))

		idx_train = dataMain['idx_train']
		idx_valid = dataMain['idx_valid']
		idx_pool = dataMain['idx_pool']

		kwargs = {'num_workers':1, 'pin_memory': True}
		transform = transforms.Compose(
					[transforms.ToTensor()])

		meanSVHN = [0.5, 0.5, 0.5]
		stdSVHN = [0.5, 0.5, 0.5]

		trainset = datasets.SVHN(root='../data', split = 'train',
											download=True, transform=transform)
		train_loader = torch.utils.data.DataLoader(trainset, batch_size=1024,
												shuffle=False, **kwargs)

		testset = datasets.SVHN(root='../data', split = 'test',
											download=True, transform= transforms.Compose(
					[transforms.ToTensor()]))
		test_loader = torch.utils.data.DataLoader(testset, batch_size=1024,
												shuffle=False, **kwargs)

		train_data_all = train_loader.dataset.data
		train_target_all = train_loader.dataset.labels

		train_target_all = np.array(train_target_all)

		train_data = train_data_all[idx_train,:,:,:]
		train_target = train_target_all[idx_train]

		val_data = train_data_all[idx_valid,:,:,:]
		val_target = train_target_all[idx_valid]

		pool_data = train_data_all[idx_pool,:,:,:]
		pool_target = train_target_all[idx_pool]

		train_data = train_data.astype('float32')
		val_data = val_data.astype('float32')
		pool_data = pool_data.astype('float32')
		
		train_data /= 255
		val_data /= 255
		pool_data /= 255

		train_data -= np.array(meanSVHN).reshape((1,3,1,1))
		train_data /= np.array(stdSVHN).reshape((1,3,1,1))

		val_data -= np.array(meanSVHN).reshape((1,3,1,1))
		val_data /= np.array(stdSVHN).reshape((1,3,1,1))

		pool_data -= np.array(meanSVHN).reshape((1,3,1,1))
		pool_data /= np.array(stdSVHN).reshape((1,3,1,1))

		train_data, train_target = torch.from_numpy(train_data), torch.from_numpy(train_target)
		val_data, val_target = torch.from_numpy(val_data), torch.from_numpy(val_target)
		pool_data, pool_target = torch.from_numpy(pool_data), torch.from_numpy(pool_target)

		test_data = test_loader.dataset.data
		test_target = test_loader.dataset.labels

		test_target = np.array(test_target)

		test_data = test_data.astype('float32')
		test_data /= 255

		test_data -= np.array(meanSVHN).reshape((1,3,1,1))
		test_data /= np.array(stdSVHN).reshape((1,3,1,1))

		test_data, test_target = torch.from_numpy(test_data), torch.from_numpy(test_target)

		return (train_data, train_target, 
				val_data, val_target,
				test_data, test_target,
				pool_data, pool_target,
				dataMain['batch_size'], dataMain['shuffleBatches'],
				dataMain['nb_epoch']
				)

	elif returnType == 'model':
		# model, _ = getCifar10Model(500, 10)
		model, _ = getSVHNModel(500, 10)
		PATH_curr = '../tempTorch/SVHN_Init_model_Experiment_evenSplit_' + np.str(e) + '.pt'
		model.load_state_dict(torch.load(PATH_curr))
		return model


def getSVHNData(trainInitSize = 200):

	kwargs = {'num_workers':1, 'pin_memory': True}
	transform = transforms.Compose(
				[transforms.ToTensor()])

	meanSVHN = [0.5, 0.5, 0.5]
	stdSVHN = [0.5, 0.5, 0.5]

	trainset = datasets.SVHN(root='../data', split = 'train',
										download=True, transform=transform)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=1024,
											shuffle=False, **kwargs)

	testset = datasets.SVHN(root='../data', split = 'test',
										download=True, transform= transforms.Compose(
				[transforms.ToTensor()]))
	test_loader = torch.utils.data.DataLoader(testset, batch_size=1024,
											shuffle=False, **kwargs)

	train_data_all = train_loader.dataset.data
	train_target_all = train_loader.dataset.labels
#########################
	train_target_all = np.array(train_target_all)

	# print(train_data[0,:,:,:])

	# valid_data = []
	# valid_target = []
	# train_data = []
	# train_target = []

	# numEachClass = int(500)

	# for i in range(0,10):
	# 	arr = np.array(np.where(train_target_all==i))
	# 	idx = np.random.permutation(arr)
	# 	idx_valid = idx[0][:numEachClass]
	# 	idx_train = idx[0][numEachClass:]
	# 	data_i =  train_data_all[ idx_valid, :,:,: ]
	# 	target_i = train_target_all[idx_valid]
	# 	valid_data.append(data_i)
	# 	valid_target.append(target_i)

	# 	data_i = train_data_all[ idx_train, :,:,: ]
	# 	target_i = train_target_all[idx_train]
	# 	train_data.append(data_i)
	# 	train_target.append(target_i)

	# train_data_all = np.concatenate(train_data, axis = 0).astype("float32")
	# train_target_all = np.concatenate(train_target, axis=0)

	# val_data = np.concatenate(valid_data, axis = 0).astype("float32")
	# val_target = np.concatenate(valid_target, axis=0)

	train_data_all, val_data, train_target_all, val_target = train_test_split(train_data_all, train_target_all, test_size = 0.1, random_state = 7)
	################
	# val_data = train_data_all[45000:50000,:,:,:]
	# val_target = train_target_all[45000:50000]
	# train_data_all = train_data_all[:45000,:,:,:]
	# train_target_all = np.array(train_target_all[:45000])

	train_data = []
	train_target = []
	pool_data = []
	pool_target = []

	numInit = np.floor(trainInitSize/10)*10
	numEachClass = int(numInit/10)

	for i in range(0,10):
		arr = np.array(np.where(train_target_all==i))
		idx = np.random.permutation(arr)
		idx_train = idx[0][:numEachClass]
		idx_pool = idx[0][numEachClass:]
		data_i =  train_data_all[ idx_train, :,:,: ]
		target_i = train_target_all[idx_train]
		train_data.append(data_i)
		train_target.append(target_i)

		data_i = train_data_all[ idx_pool, :,:,: ]
		target_i = train_target_all[idx_pool]
		pool_data.append(data_i)
		pool_target.append(target_i)

	train_data = np.concatenate(train_data, axis = 0).astype("float32")
	train_target = np.concatenate(train_target, axis=0)

	pool_data = np.concatenate(pool_data, axis = 0).astype("float32")
	pool_target = np.concatenate(pool_target, axis=0)

	# pool_data = train_data_all[6000:45000,:,:,:]
	# pool_target = train_target_all[6000:45000]

	train_target = np.array(train_target)
	val_target = np.array(val_target)
	pool_target = np.array(pool_target)

	train_data = train_data.astype('float32')
	val_data = val_data.astype('float32')
	pool_data = pool_data.astype('float32')
	
	train_data /= 255
	val_data /= 255
	pool_data /= 255

	train_data -= np.array(meanSVHN).reshape((1,3,1,1))
	train_data /= np.array(stdSVHN).reshape((1,3,1,1))

	val_data -= np.array(meanSVHN).reshape((1,3,1,1))
	val_data /= np.array(stdSVHN).reshape((1,3,1,1))

	pool_data -= np.array(meanSVHN).reshape((1,3,1,1))
	pool_data /= np.array(stdSVHN).reshape((1,3,1,1))

	train_data, train_target = torch.from_numpy(train_data), torch.from_numpy(train_target)
	val_data, val_target = torch.from_numpy(val_data), torch.from_numpy(val_target)
	pool_data, pool_target = torch.from_numpy(pool_data), torch.from_numpy(pool_target)

	test_data = test_loader.dataset.data
	test_target = test_loader.dataset.labels

	test_target = np.array(test_target)

	test_data = test_data.astype('float32')
	test_data /= 255

	test_data -= np.array(meanSVHN).reshape((1,3,1,1))
	test_data /= np.array(stdSVHN).reshape((1,3,1,1))

	test_data, test_target = torch.from_numpy(test_data), torch.from_numpy(test_target)

	return train_data, train_target, val_data, val_target, test_data, test_target, pool_data, pool_target, 10


def getCIFAR10Data(trainInitSize = 200):

	kwargs = {'num_workers':1, 'pin_memory': True}
	transform = transforms.Compose(
				[transforms.ToTensor()])

	meanCifar10 = [0.485, 0.456, 0.406]
	stdCifar10 = [0.229, 0.224, 0.225]

	trainset = datasets.CIFAR10(root='../data', train=True,
										download=True, transform=transform)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=1024,
											shuffle=False, **kwargs)

	testset = datasets.CIFAR10(root='../data', train=False,
										download=True, transform=transform)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=1024,
											shuffle=False, **kwargs)

	train_data_all = train_loader.dataset.data
	train_target_all = train_loader.dataset.targets
#########################
	train_target_all = np.array(train_target_all)

	# print(train_data[0,:,:,:])

	valid_data = []
	valid_target = []
	train_data = []
	train_target = []

	numEachClass = int(500)

	for i in range(0,10):
		arr = np.array(np.where(train_target_all==i))
		idx = np.random.permutation(arr)
		idx_valid = idx[0][:numEachClass]
		idx_train = idx[0][numEachClass:]
		data_i =  train_data_all[ idx_valid, :,:,: ]
		target_i = train_target_all[idx_valid]
		valid_data.append(data_i)
		valid_target.append(target_i)

		data_i = train_data_all[ idx_train, :,:,: ]
		target_i = train_target_all[idx_train]
		train_data.append(data_i)
		train_target.append(target_i)

	train_data_all = np.concatenate(train_data, axis = 0).astype("float32")
	train_target_all = np.concatenate(train_target, axis=0)

	val_data = np.concatenate(valid_data, axis = 0).astype("float32")
	val_target = np.concatenate(valid_target, axis=0)

	################
	# val_data = train_data_all[45000:50000,:,:,:]
	# val_target = train_target_all[45000:50000]
	# train_data_all = train_data_all[:45000,:,:,:]
	# train_target_all = np.array(train_target_all[:45000])

	train_data = []
	train_target = []
	pool_data = []
	pool_target = []

	numInit = np.floor(trainInitSize/10)*10
	numEachClass = int(numInit/10)

	for i in range(0,10):
		arr = np.array(np.where(train_target_all==i))
		idx = np.random.permutation(arr)
		idx_train = idx[0][:numEachClass]
		idx_pool = idx[0][numEachClass:]
		data_i =  train_data_all[ idx_train, :,:,: ]
		target_i = train_target_all[idx_train]
		train_data.append(data_i)
		train_target.append(target_i)

		data_i = train_data_all[ idx_pool, :,:,: ]
		target_i = train_target_all[idx_pool]
		pool_data.append(data_i)
		pool_target.append(target_i)

	train_data = np.concatenate(train_data, axis = 0).astype("float32")
	train_target = np.concatenate(train_target, axis=0)

	pool_data = np.concatenate(pool_data, axis = 0).astype("float32")
	pool_target = np.concatenate(pool_target, axis=0)

	# pool_data = train_data_all[6000:45000,:,:,:]
	# pool_target = train_target_all[6000:45000]

	train_data = np.rollaxis(train_data, 3, 1) #channel last to channel first
	train_target = np.array(train_target)

	val_data = np.rollaxis(val_data, 3, 1) #channel last to channel first
	val_target = np.array(val_target)

	pool_data = np.rollaxis(pool_data, 3, 1) #channel last to channel first
	pool_target = np.array(pool_target)

	train_data = train_data.astype('float32')
	val_data = val_data.astype('float32')
	pool_data = pool_data.astype('float32')
	
	train_data /= 255
	val_data /= 255
	pool_data /= 255

	train_data -= np.array(meanCifar10).reshape((1,3,1,1))
	train_data /= np.array(stdCifar10).reshape((1,3,1,1))

	val_data -= np.array(meanCifar10).reshape((1,3,1,1))
	val_data /= np.array(stdCifar10).reshape((1,3,1,1))

	pool_data -= np.array(meanCifar10).reshape((1,3,1,1))
	pool_data /= np.array(stdCifar10).reshape((1,3,1,1))

	train_data, train_target = torch.from_numpy(train_data), torch.from_numpy(train_target)
	val_data, val_target = torch.from_numpy(val_data), torch.from_numpy(val_target)
	pool_data, pool_target = torch.from_numpy(pool_data), torch.from_numpy(pool_target)

	# train_data, train_target = torch.from_numpy(train_data/255).float(), torch.from_numpy(train_target)
	# val_data, val_target = torch.from_numpy(val_data/255).float(), torch.from_numpy(val_target)
	# pool_data, pool_target = torch.from_numpy(pool_data/255).float(), torch.from_numpy(pool_target)

	test_data = test_loader.dataset.data
	test_target = test_loader.dataset.targets

	test_data = np.rollaxis(test_data, 3, 1)
	test_target = np.array(test_target)

	test_data = test_data.astype('float32')
	test_data /= 255

	test_data -= np.array(meanCifar10).reshape((1,3,1,1))
	test_data /= np.array(stdCifar10).reshape((1,3,1,1))

	test_data, test_target = torch.from_numpy(test_data), torch.from_numpy(test_target)

	# test_data, test_target = torch.from_numpy(test_data/255).float(), torch.from_numpy(test_target)

	return train_data, train_target, val_data, val_target, test_data, test_target, pool_data, pool_target, 10


def getCIFAR10InitConditions(e = 0, returnType = 'data'):

	if returnType == 'data':
		dataMain = pk.load(open('../tempTorch/CIFAR10_Experiment_Init_Data_evenSplit_' + str(e) + '.p', 'rb'))

		idx_train = dataMain['idx_train']
		idx_valid = dataMain['idx_valid']
		idx_pool = dataMain['idx_pool']

		kwargs = {'num_workers':1, 'pin_memory': True}
		transform = transforms.Compose(
					[transforms.ToTensor()])

		meanCifar10 = [0.485, 0.456, 0.406]
		stdCifar10 = [0.229, 0.224, 0.225]

		trainset = datasets.CIFAR10(root='../data', train=True,
										download=True, transform=transform)
		train_loader = torch.utils.data.DataLoader(trainset, batch_size=1024,
												shuffle=False, **kwargs)

		testset = datasets.CIFAR10(root='../data', train=False,
											download=True, transform=transform)
		test_loader = torch.utils.data.DataLoader(testset, batch_size=1024,
												shuffle=False, **kwargs)

		train_data_all = train_loader.dataset.data
		train_target_all = train_loader.dataset.targets

		train_target_all = np.array(train_target_all)

		train_data = train_data_all[idx_train,:,:,:]
		train_target = train_target_all[idx_train]

		val_data = train_data_all[idx_valid,:,:,:]
		val_target = train_target_all[idx_valid]

		pool_data = train_data_all[idx_pool,:,:,:]
		pool_target = train_target_all[idx_pool]

		train_data = np.rollaxis(train_data, 3, 1) #channel last to channel first
		val_data = np.rollaxis(val_data, 3, 1) #channel last to channel first
		pool_data = np.rollaxis(pool_data, 3, 1) #channel last to channel first

		train_data = train_data.astype('float32')
		val_data = val_data.astype('float32')
		pool_data = pool_data.astype('float32')
		
		train_data /= 255
		val_data /= 255
		pool_data /= 255

		train_data -= np.array(meanCifar10).reshape((1,3,1,1))
		train_data /= np.array(stdCifar10).reshape((1,3,1,1))

		val_data -= np.array(meanCifar10).reshape((1,3,1,1))
		val_data /= np.array(stdCifar10).reshape((1,3,1,1))

		pool_data -= np.array(meanCifar10).reshape((1,3,1,1))
		pool_data /= np.array(stdCifar10).reshape((1,3,1,1))

		train_data, train_target = torch.from_numpy(train_data), torch.from_numpy(train_target)
		val_data, val_target = torch.from_numpy(val_data), torch.from_numpy(val_target)
		pool_data, pool_target = torch.from_numpy(pool_data), torch.from_numpy(pool_target)

		test_data = test_loader.dataset.data
		test_target = test_loader.dataset.targets

		test_target = np.array(test_target)
		test_data = np.rollaxis(test_data, 3, 1)

		test_data = test_data.astype('float32')
		test_data /= 255

		test_data -= np.array(meanCifar10).reshape((1,3,1,1))
		test_data /= np.array(stdCifar10).reshape((1,3,1,1))

		test_data, test_target = torch.from_numpy(test_data), torch.from_numpy(test_target)

		return (train_data, train_target, 
				val_data, val_target,
				test_data, test_target,
				pool_data, pool_target,
				dataMain['batch_size'], dataMain['shuffleBatches'],
				dataMain['nb_epoch']
				)

	elif returnType == 'model':
		model, _ = getCifar10Model(500, 10)
		PATH_curr = '../tempTorch/CIFAR10_Init_model_Experiment_evenSplit_' + np.str(e) + '.pt'
		model.load_state_dict(torch.load(PATH_curr))
		return model


def getCIFAR100InitConditions(e = 0, returnType = 'data'):

	if returnType == 'data':
		dataMain = pk.load(open('../tempTorch/CIFAR100_Experiment_Init_Data_evenSplit_' + str(e) + '.p', 'rb'))

		idx_train = dataMain['idx_train']
		idx_valid = dataMain['idx_valid']
		idx_pool = dataMain['idx_pool']

		kwargs = {'num_workers':1, 'pin_memory': True}
		transform = transforms.Compose(
					[transforms.ToTensor()])

		meanCifar100 = [0.485, 0.456, 0.406]
		stdCifar100 = [0.229, 0.224, 0.225]

		trainset = datasets.CIFAR100(root='../data', train=True,
										download=True, transform=transform)
		train_loader = torch.utils.data.DataLoader(trainset, batch_size=1024,
												shuffle=False, **kwargs)

		testset = datasets.CIFAR100(root='../data', train=False,
											download=True, transform=transform)
		test_loader = torch.utils.data.DataLoader(testset, batch_size=1024,
												shuffle=False, **kwargs)

		train_data_all = train_loader.dataset.data
		train_target_all = train_loader.dataset.targets

		train_target_all = np.array(train_target_all)

		train_data = train_data_all[idx_train,:,:,:]
		train_target = train_target_all[idx_train]

		val_data = train_data_all[idx_valid,:,:,:]
		val_target = train_target_all[idx_valid]

		pool_data = train_data_all[idx_pool,:,:,:]
		pool_target = train_target_all[idx_pool]

		train_data = np.rollaxis(train_data, 3, 1) #channel last to channel first
		val_data = np.rollaxis(val_data, 3, 1) #channel last to channel first
		pool_data = np.rollaxis(pool_data, 3, 1) #channel last to channel first

		train_data = train_data.astype('float32')
		val_data = val_data.astype('float32')
		pool_data = pool_data.astype('float32')
		
		train_data /= 255
		val_data /= 255
		pool_data /= 255

		train_data -= np.array(meanCifar100).reshape((1,3,1,1))
		train_data /= np.array(stdCifar100).reshape((1,3,1,1))

		val_data -= np.array(meanCifar100).reshape((1,3,1,1))
		val_data /= np.array(stdCifar100).reshape((1,3,1,1))

		pool_data -= np.array(meanCifar100).reshape((1,3,1,1))
		pool_data /= np.array(stdCifar100).reshape((1,3,1,1))

		train_data, train_target = torch.from_numpy(train_data), torch.from_numpy(train_target)
		val_data, val_target = torch.from_numpy(val_data), torch.from_numpy(val_target)
		pool_data, pool_target = torch.from_numpy(pool_data), torch.from_numpy(pool_target)

		test_data = test_loader.dataset.data
		test_target = test_loader.dataset.targets

		test_target = np.array(test_target)
		test_data = np.rollaxis(test_data, 3, 1)

		test_data = test_data.astype('float32')
		test_data /= 255

		test_data -= np.array(meanCifar100).reshape((1,3,1,1))
		test_data /= np.array(stdCifar100).reshape((1,3,1,1))

		test_data, test_target = torch.from_numpy(test_data), torch.from_numpy(test_target)

		return (train_data, train_target, 
				val_data, val_target,
				test_data, test_target,
				pool_data, pool_target,
				dataMain['batch_size'], dataMain['shuffleBatches'],
				dataMain['nb_epoch']
				)

	elif returnType == 'model':
		model, _ = getCifar100Model(500, 100)
		PATH_curr = '../tempTorch/CIFAR100_Init_model_Experiment_evenSplit_' + np.str(e) + '.pt'
		model.load_state_dict(torch.load(PATH_curr))
		return model


def getCIFAR100Data():

	kwargs = {'num_workers':1, 'pin_memory': True}
	transform = transforms.Compose(
				[transforms.ToTensor()])

	meanCifar = [0.485, 0.456, 0.406]
	stdCifar = [0.229, 0.224, 0.225]

	trainset = datasets.CIFAR100(root='../data', train=True,
										download=True, transform=transform)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=1024,
											shuffle=False, **kwargs)

	testset = datasets.CIFAR100(root='../data', train=False,
										download=True, transform=transform)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=1024,
											shuffle=False, **kwargs)

	train_data_all = train_loader.dataset.data
	train_target_all = train_loader.dataset.targets

	train_data = train_data_all[:6000,:,:,:]
	train_target = train_target_all[:6000]

	pool_data = train_data_all[6000:45000,:,:,:]
	pool_target = train_target_all[6000:45000]

	val_data = train_data_all[45000:50000,:,:,:]
	val_target = train_target_all[45000:50000]

	train_data = np.rollaxis(train_data, 3, 1) #channel last to channel first
	train_target = np.array(train_target)

	val_data = np.rollaxis(val_data, 3, 1) #channel last to channel first
	val_target = np.array(val_target)

	pool_data = np.rollaxis(pool_data, 3, 1) #channel last to channel first
	pool_target = np.array(pool_target)

	train_data = train_data.astype('float32')
	val_data = val_data.astype('float32')
	pool_data = pool_data.astype('float32')
	
	train_data /= 255
	val_data /= 255
	pool_data /= 255

	train_data -= np.array(meanCifar).reshape((1,3,1,1))
	train_data /= np.array(stdCifar).reshape((1,3,1,1))

	val_data -= np.array(meanCifar).reshape((1,3,1,1))
	val_data /= np.array(stdCifar).reshape((1,3,1,1))

	pool_data -= np.array(meanCifar).reshape((1,3,1,1))
	pool_data /= np.array(stdCifar).reshape((1,3,1,1))

	train_data, train_target = torch.from_numpy(train_data), torch.from_numpy(train_target)
	val_data, val_target = torch.from_numpy(val_data), torch.from_numpy(val_target)
	pool_data, pool_target = torch.from_numpy(pool_data), torch.from_numpy(pool_target)

	test_data = test_loader.dataset.data
	test_target = test_loader.dataset.targets

	test_data = np.rollaxis(test_data, 3, 1)
	test_target = np.array(test_target)

	test_data = test_data.astype('float32')
	test_data /= 255

	test_data -= np.array(meanCifar).reshape((1,3,1,1))
	test_data /= np.array(stdCifar).reshape((1,3,1,1))

	test_data, test_target = torch.from_numpy(test_data), torch.from_numpy(test_target)

	return train_data, train_target, val_data, val_target, test_data, test_target, pool_data, pool_target, 100