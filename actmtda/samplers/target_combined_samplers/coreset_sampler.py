import numpy as np
import torch

from actmtda.datasets import build_dataloader
from actmtda.utils.registry import SAMPLER
from actmtda.samplers.base_sampler import BaseSampler

from sklearn.metrics import pairwise_distances

import abc
from tqdm import tqdm

class SamplingMethod(object):
	__metaclass__ = abc.ABCMeta
	
	@abc.abstractmethod
	def __init__(self, X, y, seed, **kwargs):
		self.X = X
		self.y = y
		self.seed = seed
	
	def flatten_X(self):
		shape = self.X.shape
		flat_X = self.X
		if len(shape) > 2:
			flat_X = np.reshape(self.X, (shape[0],np.product(shape[1:])))
		return flat_X
	
	@abc.abstractmethod
	def select_batch_(self):
		return
	
	def select_batch(self, **kwargs):
		return self.select_batch_(**kwargs)
	
	def select_batch_unc_(self, **kwargs):
		return self.select_batch_unc_(**kwargs)
	
	def to_dict(self):
		return None

class kCenterGreedy(SamplingMethod):

	def __init__(self, X,  metric='euclidean'):
		self.X = X
		self.flat_X = self.flatten_X()
		self.name = 'kcenter'
		self.features = self.flat_X
		self.metric = metric
		self.min_distances = None
		self.max_distances = None
		self.n_obs = self.X.shape[0]
		self.already_selected = []

	def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
		"""Update min distances given cluster centers.
		Args:
		  cluster_centers: indices of cluster centers
		  only_new: only calculate distance for newly selected points and update
			min_distances.
		  rest_dist: whether to reset min_distances.
		"""

		if reset_dist:
			self.min_distances = None
		if only_new:
			cluster_centers = [d for d in cluster_centers
							   if d not in self.already_selected]
		if len(cluster_centers) > 0:
			x = self.features[cluster_centers]
			# Update min_distances for all examples given new cluster center.
			dist = pairwise_distances(self.features, x, metric=self.metric)  # ,n_jobs=4)
	
			if self.min_distances is None:
				self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
			else:
				self.min_distances = np.minimum(self.min_distances, dist)
	
	def select_batch_(self, already_selected, N, **kwargs):
		"""
		Diversity promoting active learning method that greedily forms a batch
		to minimize the maximum distance to a cluster center among all unlabeled
		datapoints.
		Args:
		  model: model with scikit-like API with decision_function implemented
		  already_selected: index of datapoints already selected
		  N: batch size
		Returns:
		  indices of points selected to minimize distance to cluster centers
		"""
		
		print('Calculating distances...')
		self.update_distances(already_selected, only_new=False, reset_dist=False)
		
		new_batch = []
		
		print("running coreset sampling ... ")
		for _ in tqdm(range(N), ncols=100):
			if self.already_selected is None:
				# Initialize centers with a randomly selected datapoint
				ind = np.random.choice(np.arange(self.n_obs))
			else:
				ind = np.argmax(self.min_distances)
			# New examples should not be in already selected since those points
			# should have min_distance of zero to a cluster center.
			assert ind not in already_selected
			
			self.update_distances([ind], only_new=True, reset_dist=False)
			new_batch.append(ind)
		print('Maximum distance from cluster centers is %0.2f'
		      % max(self.min_distances))
		
		self.already_selected = already_selected
		
		return new_batch


@SAMPLER.register("target-combined_coreset_sampler")
class TargetCombinedCoresetSampler(BaseSampler):
	def __init__(self, *args, **kwargs):
		super(TargetCombinedCoresetSampler, self).__init__(*args, **kwargs)
	
	def __call__(self, model, discriminator=None, *args, **kwargs):
		self._stage_init()
		
		unlabeled_indices = self.get_image_idx(self.unlabeled_set)
		target_dataloader = build_dataloader(self.target_dataset,
		                                     self.cfg.VAL.BATCH_SIZE,
		                                     self.cfg.VAL.NUM_WORKER,
		                                     is_train=False,
		                                     labeled_set=unlabeled_indices)
		self.logger.info("computing features ... ")
		feats = []
		with torch.no_grad():
			model.eval()
			for batch_data in tqdm(target_dataloader, ncols=100):
				images = batch_data['image'].cuda()
				feat = model.encode_feat(images).cpu().numpy()
				feats.append(feat)
		feats = np.concatenate(feats)
		
		sampler = kCenterGreedy(feats)
		selected_indices = sampler.select_batch_([], self.stage_budget)
		selected_image_ids = [self.unlabeled_set[ind] for ind in selected_indices]
		
		for image_id in selected_image_ids:
			domain = self.id2domain_mapping[image_id]
			self.stage_domain_labeled_set[domain].append(image_id)
		self.unlabeled_set = np.setdiff1d(np.array(self.unlabeled_set, dtype=np.int32), selected_image_ids).tolist()
		
		self._stage_finish()
		
		labeled_set = self.get_labeled_list_from_domain_labeled_set(self.domain_labeled_set)
		return labeled_set


