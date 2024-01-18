import numpy as np
import torch

from actmtda.datasets import build_dataloader
from actmtda.utils.registry import SAMPLER
from actmtda.samplers.base_sampler import BaseSampler

from tqdm import tqdm

@SAMPLER.register("target-combined_random_sampler")
class TargetCombinedRandomSampler(BaseSampler):
	def __init__(self, *args, **kwargs):
		super(TargetCombinedRandomSampler, self).__init__(*args, **kwargs)
	
	def __call__(self, model, discriminator=None, *args, **kwargs):
		self._stage_init()
		
		selected_image_ids = np.random.choice(self.unlabeled_set, size=self.stage_budget, replace=False).astype(np.int32).tolist()
		
		for image_id in selected_image_ids:
			domain = self.id2domain_mapping[image_id]
			self.stage_domain_labeled_set[domain].append(image_id)
		self.unlabeled_set = np.setdiff1d(np.array(self.unlabeled_set, dtype=np.int32), selected_image_ids).tolist()
		
		self._stage_finish()
		
		labeled_set = self.get_labeled_list_from_domain_labeled_set(self.domain_labeled_set)
		return labeled_set

