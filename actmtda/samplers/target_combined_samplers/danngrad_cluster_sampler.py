import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from actmtda.datasets import build_dataloader
from actmtda.utils.registry import SAMPLER
from actmtda.samplers.base_sampler import BaseSampler

from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import pdb
import math
from scipy import stats
from prettytable import PrettyTable


def pairwise_distances_memory_saving_version(X, y, device='cuda:0'):
	'''
	x: np array (m, K)
	y: np array (K,)
	'''
	pdist = nn.PairwiseDistance(p=2)
	
	m = len(X)
	n_x_split = int(math.ceil(m / 1024.))
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
		Ddist = (D2 ** 2) / sum(D2 ** 2)
		if weights is not None:
			Ddist *= weights
			Ddist /= sum(Ddist)
		customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
		ind = customDist.rvs(size=1)[0]
		mu.append(X[ind])
		indsAll.append(ind)
		cent += 1
	return indsAll


def coreset(X, K, device, weights=None):
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
		Ddist = D2 / sum(D2)
		if weights is not None:
			Ddist *= weights
			Ddist /= sum(Ddist)
		ind = np.argmax(Ddist)
		mu.append(X[ind])
		indsAll.append(ind)
		cent += 1
	return indsAll


def splitmax(X, cluster_size, K, weights):
	km = KMeans(cluster_size)
	labels = km.fit_predict(X)
	cand_indices = []
	cand_weights = []
	for i in range(cluster_size):
		incluster_indices = np.nonzero(labels==i)[0]
		incluster_weights = weights[incluster_indices]
		argmax_idx = np.argmax(incluster_weights)
		cand_idx = incluster_indices[argmax_idx]
		cand_weight = incluster_weights.max()
		cand_indices.append(cand_idx)
		cand_weights.append(cand_weight)
	topk_cluster_idx = np.argsort(np.array(cand_weights))[-K:]
	selected_idx = [cand_indices[t] for t in topk_cluster_idx]
	return selected_idx
	

@SAMPLER.register("target-combined_danngrad-cluster_sampler")
class TargetCombinedDANNGradClusterSampler(BaseSampler):
	def __init__(self, *args, **kwargs):
		super(TargetCombinedDANNGradClusterSampler, self).__init__(*args, **kwargs)
		self.apply_stage = self.cfg.SAMPLER.DANN.APPLY_STAGE
		self.is_domainwise = self.cfg.SAMPLER.DANN.IS_DOMAINWISE
		self.is_classwise = self.cfg.SAMPLER.DANN.IS_CLASSWISE
		self.agg = self.cfg.SAMPLER.DANN.AGG
		assert self.agg in [
			'cls-grad-scaleXcos',
			'cls-grad-scaleXoneplus-cos',
			
			
			'cos', 'cls-grad-scale', 'dot', 'oneplus-cos',
			'entmin-oneplus-cos', 'entmin-cls-grad-scale', 'entmin-cls-grad-scaleXoneplus-cos',
			'normW-cls-grad-scale', 'normW-cls-grad-scaleXoneplus-cos',
			'cls-grad-scaleXabs-cos', 'cls-grad-scaleXminmax-cos', 'cls-grad-scaleXoneplus-dot',
			'cls-grad-scaleXoneplus-probdiff-cos',
			'cls-grad-scaleXoneplus-allposprobdiff-cos',
			'cls-grad-scaleXoneplus-posprobdiff-cos',
			'cls-grad-scaleXoneplus-etgtprob-cos',
			'cls-grad-scaleXoneplus-clippedetgtprob-cos',
			'cls-grad-scaleXoneplus-abs-cos',
		                    ], self.agg
		self.normalization = self.cfg.SAMPLER.DANN.NORMALIZATION
		self.normalization_level = self.cfg.SAMPLER.DANN.NORMALIZATION_LEVEL
		assert self.normalization in ['max', 'minmax'], self.normalization
		assert self.normalization_level in ['all', 'domain', 'class', 'domain-class'], self.normalization_level
		self.T = self.cfg.SAMPLER.CLUE.SOFTMAX_T
		self.domain_T = self.cfg.SAMPLER.DOMAIN_T
		
		self.danngrad_weight = self.cfg.SAMPLER.DANN.DANNGRAD_WEIGHT
		self.entropy_weight = self.cfg.SAMPLER.DANN.ENTROPY_WEIGHT
		self.cluster_method = self.cfg.SAMPLER.CLUSTER_METHOD
		assert self.cluster_method in ['kmeans', 'kmeans++', 'coreset', 'splitmax']
		
		self.use_weight_scale = self.cfg.SAMPLER.USE_WEIGHT_RESCALE
		self.weight_scale_factor = self.cfg.SAMPLER.WEIGHT_RESCALE_FACTOR
		if self.cluster_method == 'splitmax':
			self.pre_cluster_size = self.cfg.SAMPLER.PRE_CLUSTER_SIZE
			assert self.pre_cluster_size >= self.stage_budget
		self.random_cand_max_num = self.cfg.SAMPLER.RANDOM_CAND_MAX_NUM
		self.indom_aux_loss_weight = self.cfg.LOSS.INDOM_AUX_LOSS_WEIGHT
	
	def danngrad_metric(self, cls_grads, dann_grads, domain_probs, domains, labels, cls_probs):
		n = len(cls_grads)
		if self.agg == 'cos':
			cls_grads = cls_grads / np.linalg.norm(cls_grads, axis=1, keepdims=True)
			dann_grads = dann_grads / np.linalg.norm(dann_grads, axis=1, keepdims=True)
			cos_sim = (cls_grads * dann_grads).sum(axis=1)
			danngrad_importance = cos_sim + 1e-8
		elif self.agg == 'dot':
			dot_sim = (cls_grads * dann_grads).sum(axis=1)
			danngrad_importance = dot_sim + 1e-8
		elif self.agg in ['oneplus-cos', 'entmin-oneplus-cos']:
			cls_grads = cls_grads / (np.linalg.norm(cls_grads, axis=1, keepdims=True) + 1e-8)
			dann_grads = dann_grads / (np.linalg.norm(dann_grads, axis=1, keepdims=True) + 1e-8)
			cos_sim = (cls_grads * dann_grads).sum(axis=1)
			dot_sim = 1+cos_sim
			danngrad_importance = dot_sim + 1e-8
		########################
		# cls-grad-scale version
		elif self.agg == 'cls-grad-scaleXcos':
			cls_grad_norms = np.linalg.norm(cls_grads, axis=1)
			cls_grads = cls_grads / np.linalg.norm(cls_grads, axis=1, keepdims=True)
			dann_grads = dann_grads / np.linalg.norm(dann_grads, axis=1, keepdims=True)
			cos_sim = (cls_grads * dann_grads).sum(axis=1)
			cos_sim = (cos_sim + 1)/2.
			dot_sim = cls_grad_norms * cos_sim
			danngrad_importance = dot_sim + 1e-8
		elif self.agg in ['cls-grad-scaleXoneplus-cos', 'entmin-cls-grad-scaleXoneplus-cos', 'normW-cls-grad-scaleXoneplus-cos']:
			cls_grad_norms = np.linalg.norm(cls_grads, axis=1)
			cls_grads = cls_grads / np.linalg.norm(cls_grads, axis=1, keepdims=True)
			dann_grads = dann_grads / np.linalg.norm(dann_grads, axis=1, keepdims=True)
			cos_sim = (cls_grads * dann_grads).sum(axis=1)
			dot_sim = cls_grad_norms * (1+cos_sim)
			danngrad_importance = dot_sim + 1e-8
		elif self.agg in 'cls-grad-scaleXoneplus-dot':
			cls_grad_norms = np.linalg.norm(cls_grads, axis=1)
			dann_grad_norms = np.linalg.norm(dann_grads, axis=1)
			dann_grad_norms /= dann_grad_norms.max()
			cls_grads = cls_grads / np.linalg.norm(cls_grads, axis=1, keepdims=True)
			dann_grads = dann_grads / np.linalg.norm(dann_grads, axis=1, keepdims=True)
			cos_sim = (cls_grads * dann_grads).sum(axis=1) * dann_grad_norms
			dot_sim = cls_grad_norms * (1+cos_sim)
			danngrad_importance = dot_sim + 1e-8
		elif self.agg in 'cls-grad-scaleXoneplus-probdiff-cos':
			cls_grad_norms = np.linalg.norm(cls_grads, axis=1)
			cls_grads = cls_grads / np.linalg.norm(cls_grads, axis=1, keepdims=True)
			dann_grads = dann_grads / np.linalg.norm(dann_grads, axis=1, keepdims=True)
			cos_sim = (cls_grads * dann_grads).sum(axis=1)
			tgt_probs = domain_probs[np.arange(n).astype(np.int32), domains]
			src_probs = domain_probs[:, 0]
			probdiffs = tgt_probs - src_probs
			probdiffs = np.clip(probdiffs, 0., 1.0)
			probdiffs /= probdiffs.max()
			dot_sim = cls_grad_norms * (1+cos_sim * probdiffs)
			danngrad_importance = dot_sim + 1e-8
		elif self.agg in 'cls-grad-scaleXoneplus-posprobdiff-cos':
			cls_grad_norms = np.linalg.norm(cls_grads, axis=1)
			cls_grads = cls_grads / np.linalg.norm(cls_grads, axis=1, keepdims=True)
			dann_grads = dann_grads / np.linalg.norm(dann_grads, axis=1, keepdims=True)
			cos_sim = (cls_grads * dann_grads).sum(axis=1)
			tgt_probs = domain_probs[np.arange(n).astype(np.int32), domains]
			src_probs = domain_probs[:, 0]
			probdiffs = tgt_probs - src_probs
			posprobdiffs = (probdiffs > 0).astype(np.float32)
			dot_sim = cls_grad_norms * (1+cos_sim * posprobdiffs)
			danngrad_importance = dot_sim + 1e-8
		elif self.agg in 'cls-grad-scaleXoneplus-allposprobdiff-cos':
			cls_grad_norms = np.linalg.norm(cls_grads, axis=1)
			cls_grads = cls_grads / np.linalg.norm(cls_grads, axis=1, keepdims=True)
			dann_grads = dann_grads / np.linalg.norm(dann_grads, axis=1, keepdims=True)
			cos_sim = (cls_grads * dann_grads).sum(axis=1)
			max_tgt_probs = domain_probs[:, 1:].max(axis=1)
			src_probs = domain_probs[:, 0]
			probdiffs = max_tgt_probs - src_probs
			posprobdiffs = (probdiffs > 0).astype(np.float32)
			dot_sim = cls_grad_norms * (1+cos_sim * posprobdiffs)
			danngrad_importance = dot_sim + 1e-8
		elif self.agg == 'cls-grad-scaleXoneplus-etgtprob-cos':
			cls_grad_norms = np.linalg.norm(cls_grads, axis=1)
			cls_grads = cls_grads / np.linalg.norm(cls_grads, axis=1, keepdims=True)
			dann_grads = dann_grads / np.linalg.norm(dann_grads, axis=1, keepdims=True)
			cos_sim = (cls_grads * dann_grads).sum(axis=1)
			etgtprobs = domain_probs[np.arange(n).astype(np.int32), domains] / (domain_probs[:, 0] + 1e-8)
			etgtprob_cos_sim = etgtprobs * cos_sim
			dot_sim = cls_grad_norms * (1+etgtprob_cos_sim)
			danngrad_importance = dot_sim + 1e-8
		elif self.agg == 'cls-grad-scaleXoneplus-clippedetgtprob-cos':
			cls_grad_norms = np.linalg.norm(cls_grads, axis=1)
			cls_grads = cls_grads / np.linalg.norm(cls_grads, axis=1, keepdims=True)
			dann_grads = dann_grads / np.linalg.norm(dann_grads, axis=1, keepdims=True)
			cos_sim = (cls_grads * dann_grads).sum(axis=1)
			etgtprobs = domain_probs[np.arange(n).astype(np.int32), domains] / (domain_probs[:, 0] + 1e-8)
			clippedetgtprobs = np.clip(etgtprobs-1., 0., None)
			clippedetgtprob_cos_sim = clippedetgtprobs * cos_sim
			dot_sim = cls_grad_norms * (1+clippedetgtprob_cos_sim)
			danngrad_importance = dot_sim + 1e-8
		elif self.agg == 'cls-grad-scaleXoneplus-abs-cos':
			cls_grad_norms = np.linalg.norm(cls_grads, axis=1)
			cls_grads = cls_grads / np.linalg.norm(cls_grads, axis=1, keepdims=True)
			dann_grads = dann_grads / np.linalg.norm(dann_grads, axis=1, keepdims=True)
			cos_sim = (cls_grads * dann_grads).sum(axis=1)
			abs_cos_sim = np.abs(cos_sim)
			dot_sim = cls_grad_norms * (1 + abs_cos_sim)
			danngrad_importance = dot_sim + 1e-8
		elif self.agg == 'cls-grad-scaleXabs-cos':
			cls_grad_norms = np.linalg.norm(cls_grads, axis=1)
			cls_grads = cls_grads / np.linalg.norm(cls_grads, axis=1, keepdims=True)
			dann_grads = dann_grads / np.linalg.norm(dann_grads, axis=1, keepdims=True)
			cos_sim = (cls_grads * dann_grads).sum(axis=1)
			abs_cos_sim = np.abs(cos_sim)
			dot_sim = cls_grad_norms * abs_cos_sim
			danngrad_importance = dot_sim + 1e-8
		elif self.agg == 'cls-grad-scaleXminmax-cos':
			cls_grad_norms = np.linalg.norm(cls_grads, axis=1)
			cls_grads = cls_grads / np.linalg.norm(cls_grads, axis=1, keepdims=True)
			dann_grads = dann_grads / np.linalg.norm(dann_grads, axis=1, keepdims=True)
			cos_sim = (cls_grads * dann_grads).sum(axis=1)
			normalized_cos_sim = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min())
			dot_sim = cls_grad_norms * normalized_cos_sim
			danngrad_importance = dot_sim + 1e-8
		elif self.agg in ['cls-grad-scale', 'entmin-cls-grad-scale', 'normW-cls-grad-scale']:
			danngrad_importance = np.linalg.norm(cls_grads, axis=1) + 1e-8
		else:
			raise Exception()
		
		return danngrad_importance
	
	def norm_danngrad_importance(self, danngrad_importance):
		if self.normalization == 'max':
			danngrad_importance = danngrad_importance / danngrad_importance.max()
		elif self.normalization == 'minmax':
			danngrad_importance = (danngrad_importance-danngrad_importance.min()) / (danngrad_importance.max()-danngrad_importance.min())
		else: raise Exception()
		
		# assert np.all(danngrad_importance>=0), danngrad_importance.min()
		return danngrad_importance
	
	def _compute_domainwise_classwise_sample_weights(self, cls_grads, dann_grads, domain_probs, domains, labels, cls_probs):
		n = len(cls_grads)
		danngrad_importance = np.zeros(n)
	
		for domain_idx in range(1, self.n_target+1):
			indomain_sample_idx = np.nonzero(domains==domain_idx)[0]
			for cls_idx in range(self.num_class):
				inclass_sample_idx = np.nonzero(labels==cls_idx)[0]
				indomainclass_sample_idx = np.intersect1d(indomain_sample_idx, inclass_sample_idx)
				
				indomainclass_sample_danngrad_importance = self.danngrad_metric(cls_grads[indomainclass_sample_idx],
				                                                                dann_grads[indomainclass_sample_idx],
				                                                                domain_probs[indomainclass_sample_idx],
				                                                                domains[indomainclass_sample_idx],
				                                                                labels[indomainclass_sample_idx],
				                                                                cls_probs[indomainclass_sample_idx])
				if "class" in self.normalization_level:
					indomainclass_sample_danngrad_importance = self.norm_danngrad_importance(indomainclass_sample_danngrad_importance)
				danngrad_importance[indomainclass_sample_idx] = indomainclass_sample_danngrad_importance
			if 'domain' in self.normalization_level:
				danngrad_importance[indomain_sample_idx] = self.norm_danngrad_importance(danngrad_importance[indomain_sample_idx])
		print("danngrad raw stats ", danngrad_importance.min(), danngrad_importance.mean(), danngrad_importance.max(), danngrad_importance.std())
		
		if 'all' in self.normalization_level:
			danngrad_importance = self.norm_danngrad_importance(danngrad_importance)
		
		return danngrad_importance
	
	def _compute_domainwise_sample_weights(self, cls_grads, dann_grads, domain_probs, domains, labels, cls_probs):
		n = len(cls_grads)
		danngrad_importance = np.zeros(n)
		
		for domain_idx in range(1, self.n_target + 1):
			indomain_sample_idx = np.nonzero(domains == domain_idx)[0]
			indomain_sample_danngrad_importance = self.danngrad_metric(cls_grads[indomain_sample_idx],
			                                                           dann_grads[indomain_sample_idx],
			                                                           domain_probs[indomain_sample_idx],
			                                                           domains[indomain_sample_idx],
			                                                           labels[indomain_sample_idx],
			                                                           cls_probs[indomain_sample_idx])
			danngrad_importance[indomain_sample_idx] = indomain_sample_danngrad_importance
			if 'domain' in self.normalization_level:
				danngrad_importance[indomain_sample_idx] = self.norm_danngrad_importance(danngrad_importance[indomain_sample_idx])
		print("danngrad raw stats ", danngrad_importance.min(), danngrad_importance.mean(), danngrad_importance.max(), danngrad_importance.std())
		
		if 'all' in self.normalization_level:
			danngrad_importance = self.norm_danngrad_importance(danngrad_importance)
		
		return danngrad_importance
	
	def _compute_classwise_sample_weights(self, cls_grads, dann_grads, domain_probs, domains, labels, cls_probs):
		n = len(cls_grads)
		danngrad_importance = np.zeros(n)
		
		for cls_idx in range(self.num_class):
			inclass_sample_idx = np.nonzero(labels == cls_idx)[0]
			
			inclass_sample_danngrad_importance = self.danngrad_metric(cls_grads[inclass_sample_idx],
			                                                          dann_grads[inclass_sample_idx],
			                                                          domain_probs[inclass_sample_idx],
			                                                          domains[inclass_sample_idx],
			                                                          labels[inclass_sample_idx],
			                                                          cls_probs[inclass_sample_idx])
			if "class" in self.normalization_level:
				inclass_sample_danngrad_importance = self.norm_danngrad_importance(inclass_sample_danngrad_importance)
			danngrad_importance[inclass_sample_idx] = inclass_sample_danngrad_importance
		print("danngrad raw stats ", danngrad_importance.min(), danngrad_importance.mean(), danngrad_importance.max(), danngrad_importance.std())

		
		if 'all' in self.normalization_level:
			danngrad_importance = self.norm_danngrad_importance(danngrad_importance)
		
		return danngrad_importance
	
	def _compute_sample_weights(self, cls_grads, dann_grads, domain_probs, domains, labels, cls_probs):
		danngrad_importance = self.danngrad_metric(cls_grads, dann_grads, domain_probs, domains, labels, cls_probs)
		print("danngrad raw stats ", danngrad_importance.min(), danngrad_importance.mean(), danngrad_importance.max(), danngrad_importance.std())

		if 'all' in self.normalization_level:
			danngrad_importance = self.norm_danngrad_importance(danngrad_importance)
		
		return danngrad_importance
	
	def compute_sample_weights(self, entropies, labels, cls_probs, domain_probs, domains, cls_grads, dann_grads):
		entropies = entropies / entropies.max()
		print("entropy stats ", entropies.min(), entropies.mean(), entropies.max(), entropies.std())
		
		if self.entropy_weight > 0:
			sample_weights = entropies * self.entropy_weight
		else:
			sample_weights = np.ones_like(entropies)
		
		if self.stage <= self.apply_stage:
			if self.is_domainwise and self.is_classwise:
				danngrad_importance = self._compute_domainwise_classwise_sample_weights(cls_grads, dann_grads, domain_probs, domains, labels, cls_probs)
			elif self.is_domainwise and not self.is_classwise:
				danngrad_importance = self._compute_domainwise_sample_weights(cls_grads, dann_grads, domain_probs, domains, labels, cls_probs)
			elif not self.is_domainwise and self.is_classwise:
				danngrad_importance = self._compute_classwise_sample_weights(cls_grads, dann_grads, domain_probs, domains, labels, cls_probs)
			elif not self.is_domainwise and not self.is_classwise:
				danngrad_importance = self._compute_sample_weights(cls_grads, dann_grads, domain_probs, domains, labels, cls_probs)
			else: raise Exception()
			
			print("danngrad stats ", danngrad_importance.min(), danngrad_importance.mean(), danngrad_importance.max(), danngrad_importance.std())
			
			sample_weights *= self.danngrad_weight * danngrad_importance
		
		sample_weights /= sample_weights.max()
		if self.use_weight_scale:
			sample_weights = sample_weights ** self.weight_scale_factor
		
		# assert np.all(np.logical_and(sample_weights>=0, sample_weights<=1)), [sample_weights.min(), sample_weights.max()]
		
		return sample_weights
	
	def kmeans_cluster(self, features, weights, unlabeled_indices, ignore_idx=None):
		if ignore_idx is not None:
			weights[ignore_idx] = 0.
		
		km = KMeans(self.stage_budget)
		km.fit(features, sample_weight=weights)
		
		# Find nearest neighbors to inferred centroids
		dists = euclidean_distances(km.cluster_centers_, features)
		if ignore_idx is not None:
			max_dist = dists.max()
			dists[:, ignore_idx] = max_dist
		
		sort_idxs = dists.argsort(axis=1)
		q_idxs = []
		ax, rem = 0, len(unlabeled_indices)
		while rem > 0:
			q_idxs.extend(list(sort_idxs[:, ax][:rem]))
			q_idxs = list(set(q_idxs))
			rem = self.stage_budget - len(q_idxs)
			ax += 1
		return q_idxs
	
	def kmeanspp_cluster(self, features, weights, unlabeled_indices):
		sampled_idx = kmeansplus(features, self.stage_budget, 'cuda:0', weights=weights)
		return sampled_idx
	
	def coreset_cluster(self, features, weights, unlabeled_indices):
		sampled_idx = coreset(features, self.stage_budget, 'cuda:0', weights=weights)
		return sampled_idx
	
	def __call__(self, model, discriminator=None, *args, **kwargs):
		self._stage_init()
		
		unlabeled_set = self.unlabeled_set
		if len(unlabeled_set) > self.random_cand_max_num:
			rand_cand_idx = np.random.permutation(len(unlabeled_set))[:self.random_cand_max_num]
			unlabeled_set = [unlabeled_set[i] for i in rand_cand_idx]
		
		unlabeled_indices = self.get_image_idx(unlabeled_set)
		target_dataloader = build_dataloader(self.target_dataset,
		                                     self.cfg.VAL.BATCH_SIZE,
		                                     self.cfg.VAL.NUM_WORKER,
		                                     is_train=False,
		                                     labeled_set=unlabeled_indices)
		
		self.logger.info("computing features ... ")
		model.eval()
		discriminator.eval()
		
		source_domain_list = []
		source_domain_prob_list = []
		source_dataloader = build_dataloader(self.source_dataset,
		                                     self.cfg.VAL.BATCH_SIZE,
		                                     self.cfg.VAL.NUM_WORKER,
		                                     is_train=False)
		for batch_data in tqdm(source_dataloader, ncols=100):
			images = batch_data['image'].cuda()
			domains = batch_data['domain'].cuda()
			with torch.no_grad():
				logits, feat = model(images)
				dom_logits = discriminator(feat)
				if dom_logits.size(1) == self.n_target+2:
					dom_prob = torch.softmax(dom_logits.detach().cpu()[:, :2], dim=1).numpy()
				else:
					dom_prob = torch.softmax(dom_logits.detach().cpu(), dim=1).numpy()
			source_domain_list.append(domains.detach().cpu().numpy())
			source_domain_prob_list.append(dom_prob)
		source_domains = np.concatenate(source_domain_list)
		source_domain_probs = np.concatenate(source_domain_prob_list)
		
		feature_list = []
		cls_grad_list = []
		dann_grad_list = []
		pseudo_label_list = []
		cls_prob_list = []
		domain_list = []
		domain_prob_list = []
		entropy_list = []
		for batch_data in tqdm(target_dataloader, ncols=100):
			images = batch_data['image'].cuda()
			domains = batch_data['domain'].cuda()
			with torch.no_grad():
				feat = model.encode_feat(images)
			feat.requires_grad_(True)
			# compute cls grad
			logits = model.head(feat)
			label = logits.argmax(dim=1)
			if 'entmin' in self.agg:
				loss = torch.special.entr(torch.softmax(logits, dim=1)).sum()
			else:
				loss = F.cross_entropy(logits, label, reduction="sum")
			cls_grads = torch.autograd.grad(loss, feat)[0].detach().cpu().numpy()
			# compute cls prob
			cls_scores = torch.softmax(logits.detach().cpu() / self.T, dim=1) + 1e-8
			cls_prob_list.append(cls_scores.detach().cpu().numpy())
			entropy = torch.special.entr(cls_scores).sum(1).detach().cpu().numpy()
			# compute dom grad
			feat = feat.detach()
			feat.requires_grad_(True)
			dom_logits = discriminator(feat) / self.domain_T
			src_domains = torch.zeros_like(domains)
			if dom_logits.size(1) == self.n_target + 2:
				loss1 = F.cross_entropy(dom_logits[:, :2], src_domains, reduction='sum')
				tgt_aligned_dom_logits = torch.cat([dom_logits[:, 0:1], dom_logits[:, 2:]], dim=1)
				loss2 = self.indom_aux_loss_weight * F.cross_entropy(tgt_aligned_dom_logits, src_domains, reduction='sum')
				loss = loss1 + loss2
				dom_prob = torch.softmax(dom_logits[:, :2].detach().cpu(), dim=1).numpy()
			else:
				loss = F.cross_entropy(dom_logits, src_domains, reduction='sum')
				dom_prob = torch.softmax(dom_logits.detach().cpu(), dim=1).numpy()
			dom_grads = torch.autograd.grad(loss, feat)[0].detach().cpu().numpy()
			
			feat = feat.detach().cpu().numpy()
			
			feature_list.append(feat)
			cls_grad_list.append(cls_grads)
			dann_grad_list.append(dom_grads)
			pseudo_label_list.append(label.detach().cpu().numpy())
			
			entropy_list.append(entropy)
			domain_list.append(domains.detach().cpu().numpy())
			domain_prob_list.append(dom_prob)
			
		features = np.concatenate(feature_list)
		cls_grads = np.concatenate(cls_grad_list)
		dann_grads = np.concatenate(dann_grad_list)
		pseudo_labels = np.concatenate(pseudo_label_list)
		entropies = np.concatenate(entropy_list)
		cls_probs = np.concatenate(cls_prob_list)
		domains = np.concatenate(domain_list)
		domain_probs = np.concatenate(domain_prob_list)
		
		# print domainwise probs
		self.print_out_domainwise_probs_table(np.concatenate([source_domains, domains]), np.concatenate([source_domain_probs, domain_probs]))
		
		sample_weights = self.compute_sample_weights(entropies, pseudo_labels, cls_probs, domain_probs, domains, cls_grads, dann_grads)
		
		if self.cluster_method == 'kmeans':
			selected_idx = self.kmeans_cluster(features, sample_weights, unlabeled_indices)
		elif self.cluster_method == 'kmeans++':
			selected_idx = self.kmeanspp_cluster(features, sample_weights, unlabeled_indices)
		elif self.cluster_method == 'coreset':
			selected_idx = self.coreset_cluster(features, sample_weights, unlabeled_indices)
		elif self.cluster_method == 'splitmax':
			selected_idx = splitmax(features, self.pre_cluster_size, self.stage_budget, sample_weights)
		else:
			raise Exception()

		selected_image_ids = [unlabeled_set[ind] for ind in selected_idx]
		
		for image_id in selected_image_ids:
			domain = self.id2domain_mapping[image_id]
			self.stage_domain_labeled_set[domain].append(image_id)
		self.unlabeled_set = np.setdiff1d(np.array(self.unlabeled_set, dtype=np.int32), selected_image_ids).tolist()
		
		self._stage_finish()
		
		labeled_set = self.get_labeled_list_from_domain_labeled_set(self.domain_labeled_set)
		return labeled_set
	
	def print_out_domainwise_probs_table(self, domains, domain_probs):
		x = PrettyTable()
		
		if self.cfg.MODEL.DISCRIMINATOR.OUTPUT_DIM == len(self.cfg.DATASET.TARGET.DOMAIN_NAMES) + 1:  # v3
			domain_names = self.cfg.DATASET.SOURCE.DOMAIN_NAMES + self.cfg.DATASET.TARGET.DOMAIN_NAMES
			x.field_names = ['domain', ] + domain_names
			for domain_idx in range(self.n_target + 1):
				indomain_sample_idx = np.nonzero(domains == domain_idx)[0]
				indomain_sample_probs = domain_probs[indomain_sample_idx]
				avgs = indomain_sample_probs.mean(axis=0)
				stds = indomain_sample_probs.std(axis=0)
				x.add_row([domain_names[domain_idx], ] + [f"{avg:.2f}/{std:.2f}" for avg, std in zip(avgs, stds)])
		
		elif self.cfg.MODEL.DISCRIMINATOR.OUTPUT_DIM == 2 or \
			self.cfg.MODEL.DISCRIMINATOR.OUTPUT_DIM == 2 + len(self.cfg.DATASET.TARGET.DOMAIN_NAMES):  # v1 / d3
			domain_names = self.cfg.DATASET.SOURCE.DOMAIN_NAMES + ['-'.join(self.cfg.DATASET.TARGET.DOMAIN_NAMES), ]
			x.field_names = ['domain', ] + domain_names
			for domain_idx in range(self.n_target + 1):
				indomain_sample_idx = np.nonzero(domains == domain_idx)[0]
				indomain_sample_probs = domain_probs[indomain_sample_idx]
				avgs = indomain_sample_probs.mean(axis=0)
				stds = indomain_sample_probs.std(axis=0)
				if domain_idx == 0:
					domain_name_ = self.cfg.DATASET.SOURCE.DOMAIN_NAMES[0]
				else:
					domain_name_ = '-'.join(self.cfg.DATASET.TARGET.DOMAIN_NAMES)
				x.add_row([domain_name_, ] + [f"{avg:.2f}/{std:.2f}" for avg, std in zip(avgs, stds)])
		
		elif self.cfg.MODEL.DISCRIMINATOR.OUTPUT_DIM == 2 * len(self.cfg.DATASET.TARGET.DOMAIN_NAMES):  # v2
			domain_names = self.cfg.DATASET.TARGET.DOMAIN_NAMES
			x.field_names = ['domain', ] + domain_names
			for domain_idx in range(self.n_target + 1):
				indomain_sample_idx = np.nonzero(domains == domain_idx)[0]
				indomain_sample_probs = domain_probs[indomain_sample_idx]
				avgs = indomain_sample_probs.mean(axis=0)[1::2]
				stds = indomain_sample_probs.std(axis=0)[1::2]
				x.add_row([domain_names[domain_idx], ] + [f"{avg:.2f}/{std:.2f}" for avg, std in zip(avgs, stds)])
		
		print(x)
			
		