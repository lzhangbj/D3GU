import numpy as np
import torch

from actmtda.datasets import build_dataloader
from actmtda.utils.registry import SAMPLER
from actmtda.samplers.base_sampler import BaseSampler

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm


@SAMPLER.register("target-combined_kmeans_sampler")
class TargetCombinedKmeansSampler(BaseSampler):
	def __init__(self, *args, **kwargs):
		super(TargetCombinedKmeansSampler, self).__init__(*args, **kwargs)
	
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
				logits, feat = model(images)
				feat = feat.cpu().numpy()
				feats.append(feat)
		feats = np.concatenate(feats)

		km = KMeans(self.stage_budget)
		km.fit(feats)
		
		# Find nearest neighbors to inferred centroids
		dists = euclidean_distances(km.cluster_centers_, feats)
		sort_idxs = dists.argsort(axis=1)
		q_idxs = []
		ax, rem = 0, len(unlabeled_indices)
		while rem > 0:
			q_idxs.extend(list(sort_idxs[:, ax][:rem]))
			q_idxs = list(set(q_idxs))
			rem = self.stage_budget - len(q_idxs)
			ax += 1
		selected_image_ids = [self.unlabeled_set[ind] for ind in q_idxs]
		
		for image_id in selected_image_ids:
			domain = self.id2domain_mapping[image_id]
			self.stage_domain_labeled_set[domain].append(image_id)
		self.unlabeled_set = np.setdiff1d(np.array(self.unlabeled_set, dtype=np.int32), selected_image_ids).tolist()
		
		self._stage_finish()
		
		labeled_set = self.get_labeled_list_from_domain_labeled_set(self.domain_labeled_set)
		return labeled_set
