import torch
import torch.nn as nn
import numpy as np
from numba import njit, prange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getNoisyTypeWriter_channelOut(y, q, epsilon):

	numSamples = np.size(y)
	rndNoise = np.random.choice(2, size=(numSamples,), p = [1-epsilon, epsilon])
	y_prime = (y + rndNoise) % q  # Fq addition
	
	return y_prime


def get_qSC_channelOut(y, q, epsilon):

	numSamples = np.size(y)
	probVec = np.concatenate((np.array([ 1-epsilon]), (epsilon / (q-1))* np.ones((q-1,))))
	rndNoise = np.random.choice(q, size=(numSamples,), p=probVec)

	y_prime = (y + rndNoise) % q # Fq addition

	return y_prime

def getReverseqSCChannel(py, epsilon, q):

	probVec = np.concatenate((np.array([ 1-epsilon]), (epsilon / (q-1))* np.ones((q-1,))))

	prob_yp_given_y = np.zeros((q,q))
	for i in range(q):
		prob_yp_given_y[i,:] = probVec
		probVec = np.roll(probVec, 1)

	prob_yp = np.reshape(py, (1, q)) @ prob_yp_given_y
	prob_yp = np.squeeze(prob_yp)

	matTemp = np.reshape(1/prob_yp, (q,1)) @ np.reshape(py, (1, q))
	prob_y_given_yp = np.multiply(prob_yp_given_y, matTemp)

	return prob_y_given_yp, prob_yp_given_y


def updateTrainAndPoolPoints(X_train, y_train, X_Pool, y_Pool, Pooled_X, Pooled_Y, x_pool_index):

	x_pool_index = x_pool_index.numpy()

	X_Pool = X_Pool.numpy()
	y_Pool = y_Pool.numpy()
	X_train = X_train.numpy()
	y_train = y_train.numpy()
												
	X_Pool = np.delete(X_Pool, (x_pool_index), axis=0)
	y_Pool = np.delete(y_Pool, (x_pool_index), axis=0)

	X_train = np.concatenate((X_train, Pooled_X), axis=0)
	y_train = np.concatenate((y_train, Pooled_Y), axis=0)

	return torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(X_Pool), torch.from_numpy(y_Pool)


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


@njit(parallel=True)
def getJS_distance(dVal):

	totalSamples = len(dVal)
	log2_p = np.log2(dVal)
	rowSum = np.sum(np.multiply(dVal, log2_p), axis=1)
	# Dij = np.reshape(rowSum, (totalSamples,1)) @ np.ones((1, totalSamples))
	# Dij -= dVal @ log2_p.T
	# Dji = np.ones((totalSamples,1)) @ np.reshape(rowSum, (1, totalSamples))
	# Dji -= log2_p @ dVal.T
	# deltaMat = (Dij + Dji)/2
	# np.fill_diagonal(deltaMat, 0)
	# deltaMat = np.sqrt(deltaMat)

	mat1 = np.reshape(rowSum, (totalSamples,1)) @ np.ones((1, totalSamples))
	matTemp = np.zeros((totalSamples, totalSamples))

	for i in prange(totalSamples):
		for j in range(i):
			matTemp[i, j] = np.sum(np.multiply((dVal[i,:] + dVal[j,:]), np.log2((dVal[i,:] + dVal[j,:])/2)))
	deltaMat = (mat1 + mat1.T - matTemp - matTemp.T)/2
	np.fill_diagonal(deltaMat, 0)

	return np.sqrt(deltaMat)

def get_symmetric_kl_distance(dVal):

	totalSamples = len(dVal)
	log2_p = np.log2(dVal)
	rowSum = np.sum(np.multiply(dVal, log2_p), axis=1)
	Dij = np.reshape(rowSum, (totalSamples,1)) @ np.ones((1, totalSamples))
	Dij -= dVal @ log2_p.T
	Dji = np.ones((totalSamples,1)) @ np.reshape(rowSum, (1, totalSamples))
	Dji -= log2_p @ dVal.T
	deltaMat = (Dij + Dji)/2
	np.fill_diagonal(deltaMat, 0)
	return np.sqrt(deltaMat)


def evalModel(data_loader, model, verbose=0, stochastic_pass = True,
					 compute_metrics=True, activationName = None):

	global activation
	#   in evaluation phase, we never bother about the final noise layer (if it exist)
	if stochastic_pass:
		model.train()
	else:
		model.eval()

	test_loss = 0
	predictions = []
	activations = []
	correct = 0
	with torch.no_grad():
		for data, target in data_loader:

			data, target = data.to(device), target.to(device)
			if activationName is 'beforeNoise':
				# outputTemp = model(data)
				# output = activation[activationName]
				_, output, penultimateOut = model(data)  # penultimate output is useless variable here
				# print(output)
			else:
				output, penultimateOut = model(data)
				# output, (x_dense, x_dropO) = model(data)
				# print('x_dense = ', x_dense)
				# print('x_dropO = ', x_dropO)

			if compute_metrics:
				predictionClasses = output.argmax(dim=1, keepdim=True)
				correct += predictionClasses.eq(target.view_as(predictionClasses)).sum().item()
				# if activationName is 'beforeNoise':
				# else:
				criteria = nn.CrossEntropyLoss().cuda()
					# loss = criteria(output, target)
				test_loss += criteria(output, target).sum().item()
			else:
				activations.extend(penultimateOut.cpu().numpy())
				# activations.extend(output.cpu().numpy())
				softmaxed = F.softmax(output.cpu(), dim=1)
				predictions.extend(softmaxed.data.numpy())
				# print(softmaxed)

	if compute_metrics:
		return test_loss, correct
	else:
		return predictions, activations


