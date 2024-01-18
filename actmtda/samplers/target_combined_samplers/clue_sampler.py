import numpy as np
import torch

from actmtda.datasets import build_dataloader
from actmtda.utils.registry import SAMPLER
from actmtda.samplers.base_sampler import BaseSampler

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm


@SAMPLER.register("target-combined_clue_sampler")
class TargetCombinedClueSampler(BaseSampler):
	def __init__(self, *args, **kwargs):
		super(TargetCombinedClueSampler, self).__init__(*args, **kwargs)
		self.T = self.cfg.SAMPLER.CLUE.SOFTMAX_T
		self.select_from_domain = self.cfg.SAMPLER.SELECT_FROM_DOMAIN
		self.random_cand_max_num = self.cfg.SAMPLER.RANDOM_CAND_MAX_NUM
	
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
		feats = []
		domains = []
		sample_weights = []
		with torch.no_grad():
			model.eval()
			for batch_data in tqdm(target_dataloader, ncols=100):
				images = batch_data['image'].cuda()
				logits, feat = model(images)
				feat = feat.cpu().numpy()
				scores = torch.softmax(logits / self.T, dim=1) + 1e-8
				sample_weight = -(scores * torch.log(scores)).sum(1).cpu().numpy()
				
				feats.append(feat)
				sample_weights.append(sample_weight)
				domains.append(batch_data['domain'].detach().cpu().numpy())
		feats = np.concatenate(feats)
		sample_weights = np.concatenate(sample_weights)
		domains = np.concatenate(domains)
		if self.select_from_domain > 0:
			indomain_sample_idx = np.nonzero(domains==self.select_from_domain)[0]
			feats = feats[indomain_sample_idx]
			sample_weights = sample_weights[indomain_sample_idx]
		
		self.logger.info("kmeans ... ")
		km = KMeans(self.stage_budget)
		km.fit(feats, sample_weight=sample_weights)
		
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
		if self.select_from_domain > 0:
			q_idxs = [indomain_sample_idx[q] for q in q_idxs]
		selected_image_ids = [unlabeled_set[ind] for ind in q_idxs]
		
		for image_id in selected_image_ids:
			domain = self.id2domain_mapping[image_id]
			self.stage_domain_labeled_set[domain].append(image_id)
		self.unlabeled_set = np.setdiff1d(np.array(self.unlabeled_set, dtype=np.int32), selected_image_ids).tolist()
		
		self._stage_finish()
		
		labeled_set = self.get_labeled_list_from_domain_labeled_set(self.domain_labeled_set)
		return labeled_set
