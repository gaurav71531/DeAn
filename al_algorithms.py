import torch
import torch.utils.data as data_utils
import numpy as np
import scipy as sp
from scipy.special import softmax
from numba import njit, prange

from sklearn.cluster import AgglomerativeClustering

from utils import *

def run_k_center_greedyRoutine(deltaMat, s0, totalSamples, b):

	ctr = 0
	s = list(s0)
	numTrain = len(s0)
	min_s_ind_old = []
	n_minus_s = list(range(len(s0), totalSamples))
	while (len(s) < len(s0) + b):
		if ctr>0:
			# min_s_ind_new = np.copy(min_s_ind)
			min_s_ind_new = np.delete(min_s_ind, u)
			min_s_val_new = np.delete(min_s_val, u)
			new_min_ind = min_s_val_new < deltaMat[n_minus_s, s[-1]]
			min_s_ind = np.where(new_min_ind, min_s_ind_new, len(s)-1)
		else:
			min_s_ind = np.argmin(deltaMat[np.ix_(n_minus_s, s)], axis=1) #np.ix_ for extracting non-square matrix

		min_s_ind_fin = [s[i] for i in min_s_ind]
		min_s_val = deltaMat[n_minus_s, min_s_ind_fin]
		u = np.argmax(deltaMat[n_minus_s, min_s_ind_fin])
		s.append(n_minus_s[u])
		n_minus_s = np.delete(n_minus_s, u)
		ctr += 1

	s_minus_s0 = np.array([i for i in s if i not in s0])

	s_minus_s0_shifted = s_minus_s0-numTrain # to bring them to begin from 0 -> num_pool_subset
	# s_minus_s0_shifted = torch.from_numpy(s_minus_s0_shifted)

	return s_minus_s0_shifted
	# s_minus_s0_actual = pool_subset[s_minus_s0_shifted]


def getClusterIndices(X_train, X_Pool, model, Queries, dropout_iterations, 
						nb_classes, batch_size, params):

	if 'trainWithNoise' in params.keys():
		if params['trainWithNoise']:
			activationName = 'beforeNoise'
		else:
			activationName = None

	pool_subset_dropout = torch.from_numpy(np.arange(X_Pool.shape[0]))
	num_pool = X_Pool.shape[0]
	if 'take_pool_subset' in params.keys():
		if params['take_pool_subset']:
			if params['num_pool_subset'] < X_Pool.shape[0]:
				pool_subset_dropout = torch.from_numpy(np.random.choice(X_Pool.shape[0], params['num_pool_subset']))
				num_pool = params['num_pool_subset']
    	
	X_Pool_Dropout = X_Pool[pool_subset_dropout, :, :, :]
	y_Pool_Dropout = torch.from_numpy(np.random.randint(10, size = num_pool))

	score_All = np.zeros(shape=(X_Pool_Dropout.shape[0], nb_classes))
	All_Entropy_Dropout = np.zeros(shape=(X_Pool_Dropout.shape[0], 1))

	pool = data_utils.TensorDataset(X_Pool_Dropout, y_Pool_Dropout)
	pool_loader = data_utils.DataLoader(pool, batch_size=batch_size, shuffle=False)

	for d in range(dropout_iterations):
		dropout_score, _ = evalModel(pool_loader, model, verbose=0, 
							stochastic_pass = True, compute_metrics=False, activationName = activationName)
		dropout_score = np.array(dropout_score)
		dropout_score = np.where(dropout_score>0, dropout_score, 1e-10)
		score_All = score_All + dropout_score

		#computing F_X
		dropout_score_log = np.log2(dropout_score)
		Entropy_Compute = - np.multiply(dropout_score, dropout_score_log)
		Entropy_Per_Dropout = np.sum(Entropy_Compute, axis=1, keepdims=True)

		All_Entropy_Dropout = All_Entropy_Dropout + Entropy_Per_Dropout 

	Avg_Pi = np.divide(score_All, dropout_iterations)
	Log_Avg_Pi = np.log2(Avg_Pi)
	Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
	Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1, keepdims=True)

	Average_Entropy = np.divide(All_Entropy_Dropout, dropout_iterations)
	U_X = Entropy_Average_Pi - Average_Entropy

#  clusteting
	enlargeFactor = 2
	b = int(np.ceil(enlargeFactor * Queries))

	y_Pool_Dropout = np.array(y_Pool_Dropout).flatten()
	dVal = np.copy(Avg_Pi)
	deltaMat = getJS_distance(dVal)

	if 'val_loader' in params.keys():
		numValData = len(params['val_loader'].dataset)
		dropout_classes_All = np.zeros((numValData, dropout_iterations))
		for d in range(dropout_iterations):
			dropout_score, _ = evalModel(params['val_loader'], model, verbose=0, 
							stochastic_pass = True, compute_metrics=False, activationName = activationName)
			dropout_score = np.array(dropout_score)
			dropout_score = np.where(dropout_score>0, dropout_score, 1e-10)
			dropout_class = np.argmax(dropout_score, axis=1)
			dropout_classes_All[:,d] = dropout_class

		varRatio = np.zeros((numValData))
		for s in range(numValData):
			_, ModeFreq = sp.stats.mode(dropout_classes_All[s,:])
			varRatio[s] = 1 - ModeFreq/float(dropout_iterations)

		meanVarRatio = np.mean(varRatio)
		print('mean var ratio = %f'%(meanVarRatio))
		# ex.log_scalar("varRatio", meanVarRatio)
		if 'beta_fn' in params.keys():
			beta = params['beta_fn'](meanVarRatio)
		else:
			beta = (np.exp(1/meanVarRatio) - 1)/4

	print('beta = %f'%(beta))
	midNumClusters = params['midNumClusters']
	if midNumClusters < np.size(y_Pool_Dropout):
		clustering_middle = AgglomerativeClustering(n_clusters = midNumClusters,
									affinity = 'precomputed', linkage = 'single').fit(deltaMat)

		midClusterLabels = clustering_middle.labels_

		print(np.sort(np.bincount(midClusterLabels)))

		midCluster_median_score = np.zeros((midNumClusters,))
		midCluster_median_ind = np.zeros((midNumClusters,), dtype=int)
		for i in range(midNumClusters):
			cluster_membership = np.where(midClusterLabels == i)[0]
			U_X_cluster = U_X[cluster_membership].flatten()
			localsortedInd = U_X_cluster.argsort()[np.size(U_X_cluster)//2]
			midCluster_median_score[i] = U_X_cluster[localsortedInd]
			midCluster_median_ind[i] = cluster_membership[localsortedInd]

		reqNumSamples = midNumClusters
		arg_cluster_maxScore = (midCluster_median_score.flatten()).argsort()[-reqNumSamples:][::-1]
		
		samples_for_final_clusting = []
		ctr = 0

		midClusterLabels_not_selected = np.ones((np.size(midClusterLabels),))
		empty_clusters = []
		probVec = midCluster_median_score / np.sum(midCluster_median_score)

		print(np.sort(probVec))
		ppctr = 0
		while(True):
			while(True):
				randomInd = np.random.choice(midNumClusters, 1, p = probVec)[0]
				if randomInd not in empty_clusters:
					break

			probVec_next = -beta * deltaMat[midCluster_median_ind[randomInd],:].flatten()
			probVec_next = softmax(probVec_next)

			if ppctr < 3:
				print(np.sort(probVec_next))
				ppctr += 1

			ctr = 0
			selection_complete = False
			while(True):
				randomInd_next = np.random.choice(np.size(U_X), 1, p = probVec_next)[0]
				# selectedInd = cluster_membership[randomInd_next]
				# print('possible cluster Ind = ', cluster_membership)
				# print('si = ', selectedInd)
				if randomInd_next not in samples_for_final_clusting:
					selection_complete = True
					break
				if ctr > 20:
					# print('unable to find sample, process continue...')
					empty_clusters.append(randomInd)
					break
				ctr += 1

			if selection_complete:
				samples_for_final_clusting.append(randomInd_next)

			if len(samples_for_final_clusting) == Queries:
				break
	else:
		# samples_for_final_clusting = np.arange(np.size(pool_subset_dropout))
		samples_for_final_clusting = []
		scoreUse = U_X.flatten()
		probVec = scoreUse / np.sum(scoreUse)
		while(True):
			while(True):
				randomInd = np.random.choice(np.size(scoreUse), 1, p = probVec)[0]
				if randomInd not in samples_for_final_clusting:
					break
			samples_for_final_clusting.append(randomInd)
			if len(samples_for_final_clusting) == Queries:
				break

	x_pool_index = samples_for_final_clusting[:Queries]
	return pool_subset_dropout[x_pool_index], meanVarRatio


def getRandomIndices(X_Pool, model, Queries, params):

	# num_pool_subset = 2000

	pool_subset = np.arange(X_Pool.shape[0])
	randInd = np.random.choice(np.size(pool_subset), Queries)

	dropout_iterations = params['dropout_iterations']
	if 'val_loader' in params.keys():
		dropout_score_All = np.zeros((5000, 10, dropout_iterations))
		for d in range(dropout_iterations):
			dropout_score, _ = evalModel(params['val_loader'], model, verbose=0, 
							stochastic_pass = True, compute_metrics=False)
			dropout_score = np.array(dropout_score)
			dropout_score = np.where(dropout_score>0, dropout_score, 1e-10)
			dropout_score_All[:,:,d] = dropout_score
		
		mean_score = np.mean(dropout_score_All, axis=2)
		mean_score_add = np.sum(np.multiply(mean_score, mean_score), axis=1)
		variance_score = np.sum(np.multiply(dropout_score_All, dropout_score_All), axis=(1,2))
		variance_score = np.divide(variance_score, dropout_iterations)
		variance_score -= mean_score_add
		variance_score = np.mean(variance_score.flatten())
		beta = np.log(0.25 * 10 / variance_score)
		# ex.log_scalar("uncertainty_variance", variance_score)

		print('variance = ', variance_score, ': beta = ', beta)
		# ex.log_scalar("uncertainty_variance", variance_score)

	return torch.from_numpy(pool_subset[randInd])


def getVAALIndices(pool_loader, vae, discriminator, Queries, nb_classes, batch_size, params):

	all_preds = []
	all_indices = []

	activationName = None
	if 'trainWithNoise' in params.keys():
		if params['trainWithNoise']:
			activationName = 'beforeNoise'

	with torch.no_grad():
		for images, _ in pool_loader:
			images = images.to(device)

			_, _, mu, _ = vae(images)
			preds = discriminator(mu)

			preds = preds.cpu().data
			all_preds.extend(preds)
			# all_indices.extend(indices) 


	all_preds = torch.stack(all_preds)
	all_preds = all_preds.view(-1)
	# need to multiply by -1 to be able to use torch.topk 
	all_preds *= -1

	# select the points which the discriminator things are the most likely to be unlabeled
	_, querry_indices = torch.topk(all_preds, int(Queries))

	# print(querry_indices)
	# querry_pool_indices = np.asarray(all_indices)[querry_indices]
	# print(pool_subset[querry_indices])
	return querry_indices


def getMaxEntropyIndices(X_Pool, model, Queries, nb_classes, batch_size, params):
	#take subset of Pool Points for Test Time Dropout 
	#and do acquisition from there
	# pool_subset = 2000

	activationName = None
	if 'trainWithNoise' in params.keys():
		if params['trainWithNoise']:
			activationName = 'beforeNoise'

	pool_subset = torch.from_numpy(np.arange(X_Pool.shape[0]))
	num_pool = X_Pool.shape[0]
	if 'take_pool_subset' in params.keys():
		if params['take_pool_subset']:
			if params['num_pool_subset'] < X_Pool.shape[0]:
				pool_subset = torch.from_numpy(np.random.choice(X_Pool.shape[0], params['num_pool_subset']))
				num_pool = params['num_pool_subset']
    	
	X_Pool_subset = X_Pool[pool_subset, :, :, :]
	y_Pool_subset = torch.from_numpy(np.random.randint(10, size = num_pool))

	# score_All = np.zeros(shape=(X_Pool_subset.shape[0], nb_classes))

	pool = data_utils.TensorDataset(X_Pool_subset, y_Pool_subset)
	pool_loader = data_utils.DataLoader(pool, batch_size=batch_size, shuffle=False)

	Pi_All, _ = evalModel(pool_loader, model, verbose=0, 
							stochastic_pass = False, compute_metrics=False,
							activationName = activationName)

	# Avg_Pi = np.divide(score_All, dropout_iterations)
	Log_Pi = np.log2(Pi_All)
	Entropy_Pi = - np.multiply(Pi_All, Log_Pi)
	Entropy_Average_Pi = np.sum(Entropy_Pi, axis=1)

	U_X = Entropy_Average_Pi
	a_1d = U_X.flatten()
	x_pool_index = a_1d.argsort()[-Queries:][::-1]

	x_pool_index = torch.from_numpy(np.ascontiguousarray(x_pool_index))

	return pool_subset[x_pool_index]


def getBALDIndices(X_Pool, model, Queries, dropout_iterations, nb_classes, batch_size, params):
	#take subset of Pool Points for Test Time Dropout 
	#and do acquisition from there
	# pool_subset = 2000

	activationName = None
	if 'trainWithNoise' in params.keys():
		if params['trainWithNoise']:
			activationName = 'beforeNoise'

	pool_subset_dropout = torch.from_numpy(np.arange(X_Pool.shape[0]))
	num_pool = X_Pool.shape[0]
	if 'take_pool_subset' in params.keys():
		if params['take_pool_subset']:	
			if params['num_pool_subset'] < X_Pool.shape[0]:				
				pool_subset_dropout = torch.from_numpy(np.random.choice(X_Pool.shape[0], params['num_pool_subset']))
				num_pool = params['num_pool_subset']
    	
	X_Pool_Dropout = X_Pool[pool_subset_dropout, :, :, :]
	y_Pool_Dropout = torch.from_numpy(np.random.randint(10, size = num_pool))

	score_All = np.zeros(shape=(X_Pool_Dropout.shape[0], nb_classes))
	All_Entropy_Dropout = np.zeros(shape=X_Pool_Dropout.shape[0])

	pool = data_utils.TensorDataset(X_Pool_Dropout, y_Pool_Dropout)
	pool_loader = data_utils.DataLoader(pool, batch_size=batch_size, shuffle=False)

	for d in range(dropout_iterations):
		dropout_score, _ = evalModel(pool_loader, model, verbose=0, 
							stochastic_pass = True, compute_metrics=False,
							activationName = activationName)
		dropout_score = np.array(dropout_score)
		dropout_score = np.where(dropout_score>0, dropout_score, 1e-10)

		score_All = score_All + dropout_score

		#computing F_X
		dropout_score_log = np.log2(dropout_score)
		Entropy_Compute = - np.multiply(dropout_score, dropout_score_log)
		Entropy_Per_Dropout = np.sum(Entropy_Compute, axis=1)

		All_Entropy_Dropout = All_Entropy_Dropout + Entropy_Per_Dropout 


	Avg_Pi = np.divide(score_All, dropout_iterations)
	Log_Avg_Pi = np.log2(Avg_Pi)
	Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
	Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

	G_X = Entropy_Average_Pi

	Average_Entropy = np.divide(All_Entropy_Dropout, dropout_iterations)

	F_X = Average_Entropy

	U_X = G_X - F_X
	# U_X = 1 - F_X / G_X

	# THIS FINDS THE MINIMUM INDEX 
	# a_1d = U_X.flatten()
	# x_pool_index = a_1d.argsort()[-Queries:]

	a_1d = U_X.flatten()

	x_pool_index = a_1d.argsort()[-Queries:][::-1]

	print(np.sort(a_1d)[::-1])

	# x_pool_index = torch.from_numpy(x_pool_index)  # gives error: some of the strides of a given numpy array are negative
	# x_pool_index = torch.from_numpy(np.flip(x_pool_index, axis=0).copy())

	x_pool_index = torch.from_numpy(np.ascontiguousarray(x_pool_index))

	if 'val_loader' in params.keys():
		dropout_score_All = np.zeros((len(params['val_loader'].dataset), 10, dropout_iterations))
		for d in range(dropout_iterations):
			dropout_score, _ = evalModel(params['val_loader'], model, verbose=0, 
							stochastic_pass = True, compute_metrics=False)
			dropout_score = np.array(dropout_score)
			dropout_score = np.where(dropout_score>0, dropout_score, 1e-10)
			dropout_score_All[:,:,d] = dropout_score
		
		# variance_score = np.var(dropout_score_All, axis = 2)
		# variance_score = np.mean(variance_score.flatten())
		mean_score = np.mean(dropout_score_All, axis=2)
		mean_score_add = np.sum(np.multiply(mean_score, mean_score), axis=1)
		variance_score = np.sum(np.multiply(dropout_score_All, dropout_score_All), axis=(1,2))
		variance_score = np.divide(variance_score, dropout_iterations)
		variance_score -= mean_score_add
		variance_score = np.mean(variance_score.flatten())
		beta = np.log(0.25 / variance_score)
		print('variance = ', variance_score, ': beta = ', beta)
		# ex.log_scalar("uncertainty_variance", variance_score)


	return pool_subset_dropout[x_pool_index]


def get_k_center_greedyIndices(model, X_train, X_Pool, b, batch_size=128, params = {}):

	activationName = None
	if 'trainWithNoise' in params.keys():
		if params['trainWithNoise']:
			activationName = 'beforeNoise'

	numTrain = X_train.shape[0]
	s0 = list(range(numTrain))

	s = list(s0)

	pool_subset = torch.from_numpy(np.arange(X_Pool.shape[0]))
	totalSamples = numTrain + X_Pool.shape[0]
	if 'take_pool_subset' in params.keys():
		if params['take_pool_subset']:
			if params['num_pool_subset'] < X_Pool.shape[0]:
				pool_subset = torch.from_numpy(np.random.choice(X_Pool.shape[0], params['num_pool_subset']))
				totalSamples = numTrain + params['num_pool_subset']
	
	X_All = torch.cat((X_train, X_Pool[pool_subset,:,:,:]),dim=0)
	y_All = torch.from_numpy(np.random.randint(10, size = totalSamples))

	pool = data_utils.TensorDataset(X_All, y_All)
	pool_loader = data_utils.DataLoader(pool, batch_size=batch_size, shuffle=False)

	_, dVal = evalModel(pool_loader, model, verbose=0, 
						stochastic_pass = False, compute_metrics=False,
						activationName=activationName)

	dVal = np.array(	 dVal)
	normVec = (dVal**2) @ np.ones((dVal.shape[1],1))
	mat1 = normVec @ np.ones((1, totalSamples)) + np.ones((totalSamples,1))@normVec.T
	deltaMat = mat1 - 2 * dVal @ dVal.T
	np.fill_diagonal(deltaMat, 0)
	deltaMat = np.where(deltaMat>=0, deltaMat, 0) # to account for small numerical issues of negative values, eg. -1e-8
	deltaMat = np.sqrt(deltaMat)

	s_minus_s0_shifted = run_k_center_greedyRoutine(deltaMat, s0, totalSamples, b)
	s_minus_s0_actual = pool_subset[torch.from_numpy(s_minus_s0_shifted)]

	return s_minus_s0_actual, X_All, deltaMat


def get_k_centerIndices(deltaFn, X_train, X_Pool, b):

	numTrain = X_train.shape[0]
	s0 = list(range(numTrain))
	sgOrig, sg, X_All, deltaMat = get_k_center_greedyIndices(deltaFn, X_train, X_Pool, b)

	print('sg =', sg)

	all_index = np.arange(X_All.shape[0])

	Xi = int(np.ceil(1e-3 * len(all_index)))

	min_s_ind = []
	for i in all_index:
		min_s_ind.append(np.argmin(deltaMat[i,sg]))
	min_s_ind_fin = [sg[i] for i in min_s_ind]
	delta2_OPT = np.max(deltaMat[all_index, min_s_ind_fin])

	print('delta2_OPT=%f'%(delta2_OPT))

	lb = delta2_OPT/2
	ub = delta2_OPT

	ctr = 0
	while(True):
		if ILP_feasible_routine_Gurobi(deltaMat, Xi, (lb+ub)/2, s0, b):
			tempMat = np.where(deltaMat<= (lb+ub)/2, deltaMat, -1)
			ub = np.max(tempMat)
		else:
			tempMat = np.where(deltaMat>= (lb+ub)/2, deltaMat, np.inf)
			lb = np.min(tempMat)
		
		print(lb, ub)
		ctr += 1
		if ctr>100:break

	return [0,1]


def get_k_means_pp(deltaMat, b):

	C = []
	remainSampleInd = np.arange(len(deltaMat))
	randomInd = np.random.choice(np.size(remainSampleInd), 1)[0]
	C.append(remainSampleInd[ randomInd])
	remainSampleInd = np.delete(remainSampleInd, np.where(remainSampleInd == C[-1]))

	for t in range(1, b):
		# distMat = np.zeros((len(C), np.size(remainSampleInd)))
		# for c in range(len(C)):
		#     for k in range(np.size(remainSampleInd)):
		#         distMat[c,k] = np.linalg.norm(np.squeeze(G[C[c],:,:]) - np.squeeze(G[remainSampleInd[k],:,:]))
		# distMat = spatial.distance_matrix(G[C,:], G[remainSampleInd,:])
		# distMat = deltaMat[np.ix_(C, remainSampleInd)]
		# Dtx = np.min(distMat, axis=0)
		Dtx = np.zeros((np.size(remainSampleInd),))
		for i in range(np.size(remainSampleInd)):
			Dtx[i] = np.min(deltaMat[C,i])
		probVec = Dtx**2
		probVec /= np.sum(probVec)
		randomInd = np.random.choice(np.size(remainSampleInd), 1, p = probVec)[0]
		C.append(remainSampleInd[ randomInd])
		remainSampleInd = np.delete(remainSampleInd, np.where(remainSampleInd == C[-1]))

	return np.array(C)


def getBADGE_Indices(model, X_train, X_Pool, b, batch_size=128, params = {}):


	activationName = None
	if 'trainWithNoise' in params.keys():
		if params['trainWithNoise']:
			activationName = 'beforeNoise'

	pool_subset = torch.from_numpy(np.arange(X_Pool.shape[0]))
	if 'take_pool_subset' in params.keys():
		if params['take_pool_subset']:
			if params['num_pool_subset'] < X_Pool.shape[0]:
				pool_subset = torch.from_numpy(np.random.choice(X_Pool.shape[0], params['num_pool_subset']))
	
	pool_size = np.shape(pool_subset)[0]
	X_Pool = X_Pool[pool_subset, :, :, :]
	y_Pool = torch.from_numpy(np.random.randint(10, size = np.shape(pool_subset)[0]))

	pool = data_utils.TensorDataset(X_Pool, y_Pool)
	pool_loader = data_utils.DataLoader(pool, batch_size=batch_size, shuffle=False)

	pVal, gVal = evalModel(pool_loader, model, verbose=0, 
						stochastic_pass = False, compute_metrics=False, activationName = activationName)

	pVal = np.array(	 pVal)
	gVal = np.array(	 gVal)
	yPred = np.argmax(pVal, axis=1)
	_, gOutShape = np.shape(gVal)
	_, pOutShape = np.shape(pVal)
	gx = np.zeros((pool_size, gOutShape * pOutShape))
	# yPred_cat = np_utils.to_categorical(yPred, pOutShape)
	yPred_cat = np.eye(pOutShape)[yPred]

	for k in range(pool_size):
		preFactVec = pVal[None,k,:] - yPred_cat[None,k,:] # slicing dimension adjust for numpy
		gx[k,:] = (gVal[k,:,None] @ preFactVec).flatten()

	# deltaMat = spatial.distance_matrix(gx, gx)
	print('delta matrix started...')
	normVec = (gx**2) @ np.ones((gx.shape[1],1))
	mat1 = normVec @ np.ones((1, pool_size)) + np.ones((pool_size,1))@normVec.T
	deltaMat = mat1 - 2 * gx @ gx.T
	np.fill_diagonal(deltaMat, 0)
	deltaMat = np.where(deltaMat>=0, deltaMat, 0) # to account for small numerical issues of negative values, eg. -1e-8
	deltaMat = np.sqrt(deltaMat)
	print('delta matrix computed... shape = ', np.shape(deltaMat))
	k_means_pp_ind = get_k_means_pp(deltaMat, b)
	print('sampled...')
	return pool_subset[k_means_pp_ind]

