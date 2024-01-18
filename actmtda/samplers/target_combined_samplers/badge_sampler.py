import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from actmtda.datasets import build_dataloader
from actmtda.utils.registry import SAMPLER
from actmtda.samplers.base_sampler import BaseSampler

from tqdm import tqdm
import pdb
import math
from scipy import stats


def pairwise_distances_memory_saving_version(X, y, device='cuda:0'):
	'''
	x: np array (m, K)
	y: np array (K,)
	'''
	pdist = nn.PairwiseDistance(p=2)
	
	m = len(X)
	n_x_split = int(math.ceil(m/1024.))
	x_splits = np.array_split(X, n_x_split)
	ret_nn_dists = []
	y = torch.from_numpy(y).to(device)
	for x in x_splits:
		x = torch.from_numpy(x).to(device)
		ret_nn_dists.append(torch.flatten(pdist(x, y)).cpu().numpy())
	ret = np.concatenate(ret_nn_dists)
	return ret


def kmeansplus(X, K, device, weights=None):
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
			D2 = pairwise_distances_memory_saving_version(X, mu[-1], device)
		else:
			newD = pairwise_distances_memory_saving_version(X, mu[-1], device)
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

@SAMPLER.register("target-combined_badge_sampler")
class TargetCombinedBadgeSampler(BaseSampler):
	def __init__(self, *args, **kwargs):
		super(TargetCombinedBadgeSampler, self).__init__(*args, **kwargs)
	
	def __call__(self, model, discriminator=None, *args, **kwargs):
		self._stage_init()
		
		unlabeled_indices = self.get_image_idx(self.unlabeled_set)
		target_dataloader = build_dataloader(self.target_dataset,
		                                     self.cfg.VAL.BATCH_SIZE,
		                                     self.cfg.VAL.NUM_WORKER,
		                                     is_train=False,
		                                     labeled_set=unlabeled_indices)
		
		self.logger.info("computing features ... ")
		grad_embeddings = []
		model.eval()
		for batch_data in tqdm(target_dataloader, ncols=100):
			images = batch_data['image'].cuda()
			domain_idx = batch_data['domain'].long().numpy()
			domain_idx -= 1
			with torch.no_grad():
				feat = model.encode_feat(images)
			feat = feat.detach()
			logits = model.head(feat)
			label = logits.argmax(dim=1)
			loss = F.cross_entropy(logits, label, reduction="sum")
			l0_grads = torch.autograd.grad(loss, logits)[0].detach()
			embed_dim = feat.size(1)
			with torch.no_grad():
				# Calculate the linear layer gradients as well if needed
				l0_expand = torch.repeat_interleave(l0_grads, embed_dim, dim=1)
				l1_grads = (l0_expand * feat.repeat(1, self.num_class)).detach().cpu().numpy()
			
			# Populate embedding tensor according to the supplied argument.
			grad_embeddings.append(l1_grads)
		grad_embeddings = np.concatenate(grad_embeddings)
			
		sampled_idx = kmeansplus(grad_embeddings, self.stage_budget, 'cuda:0')
		selected_image_ids = [self.unlabeled_set[ind] for ind in sampled_idx]
		
		for image_id in selected_image_ids:
			domain = self.id2domain_mapping[image_id]
			self.stage_domain_labeled_set[domain].append(image_id)
		self.unlabeled_set = np.setdiff1d(np.array(self.unlabeled_set, dtype=np.int32), selected_image_ids).tolist()
		
		self._stage_finish()
		
		labeled_set = self.get_labeled_list_from_domain_labeled_set(self.domain_labeled_set)
		return labeled_set

	