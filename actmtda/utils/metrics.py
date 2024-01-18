import numpy as np
import torch
import math
import pdb
from scipy import stats


def pairwise_l2_distances_memory_saving(X, Y, unit=1024):
	'''
	X: np array (m, K)
	Y: np array (n, K)
	'''
	m, n = len(X), len(Y)
	n_x_split = int(math.ceil(m / unit))
	x_splits = np.array_split(X, n_x_split)
	
	n_y_split = int(math.ceil(n / unit))
	y_splits = np.array_split(Y, n_y_split)
	
	pairwise_l2_distances_list = []
	for x in x_splits:
		x = torch.from_numpy(x).cuda().unsqueeze(0)
		batch_pairwise_l2_distances_list = []
		for y in y_splits:
			y = torch.from_numpy(y).cuda().unsqueeze(0)
			pw = torch.cdist(x, y)[0].detach().cpu().numpy()
			batch_pairwise_l2_distances_list.append(pw)
		batch_pairwise_l2_distances = np.concatenate(batch_pairwise_l2_distances_list, axis=1)
		pairwise_l2_distances_list.append(batch_pairwise_l2_distances)
	pairwise_l2_distances = np.concatenate(pairwise_l2_distances_list, axis=0)
	
	return pairwise_l2_distances


def kmeansplus(X, K, weights=None):
	"""
	X: np array (N, k)
	K: number of centers
	"""
	ind = np.argmax([np.linalg.norm(s, 2) for s in X])
	mu = [X[ind]]
	indsAll = [ind]
	centInds = [0.] * len(X)
	cent = 0
	print("running kmeans ... ")
	while len(mu) < K:
		if len(mu) == 1:
			D2 = pairwise_l2_distances_memory_saving(X, mu[-1][None, :])[:, 0]
		else:
			newD = pairwise_l2_distances_memory_saving(X, mu[-1][None, :])[:, 0]
			for i in range(len(X)):
				if D2[i] > newD[i]:
					centInds[i] = cent
					D2[i] = newD[i]

		if sum(D2) == 0.0: pdb.set_trace()
		D2 = D2.ravel().astype(float)
		Ddist = (D2 ** 2)/ sum(D2 ** 2)
		if weights is not None:
			Ddist *= weights
			Ddist /= sum(Ddist)
		customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
		ind = customDist.rvs(size=1)[0]
		mu.append(X[ind])
		indsAll.append(ind)
		cent += 1
	return indsAll