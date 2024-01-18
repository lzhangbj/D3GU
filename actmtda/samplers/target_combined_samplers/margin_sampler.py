import numpy as np
import torch

from actmtda.datasets import build_dataloader
from actmtda.utils.registry import SAMPLER
from actmtda.samplers.base_sampler import BaseSampler

from tqdm import tqdm

@SAMPLER.register("target-combined_margin_sampler")
class TargetCombinedMarginSampler(BaseSampler):
	def __init__(self, *args, **kwargs):
		super(TargetCombinedMarginSampler, self).__init__(*args, **kwargs)
		self.T = self.cfg.SAMPLER.ENTROPY.SOFTMAX_T
		
	def __call__(self, model, discriminator=None, *args, **kwargs):
		self._stage_init()
		
		unlabeled_indices = self.get_image_idx(self.unlabeled_set)
		target_dataloader = build_dataloader(self.target_dataset,
		                                     self.cfg.VAL.BATCH_SIZE,
		                                     self.cfg.VAL.NUM_WORKER,
		                                     is_train=False,
		                                     labeled_set=unlabeled_indices)
		
		scores = []
		with torch.no_grad():
			model.eval()
			for batch_data in target_dataloader:
				imgs = batch_data['image'].cuda()
				logits, feats = model(imgs)
				probs = torch.softmax(logits/self.T, dim=1)
				max_probs = torch.topk(probs, 2, dim=1, sorted=True, largest=True)[0]
				entropy = (max_probs[:, 1] - max_probs[:, 0]).cpu().numpy()
				scores.append(entropy)
		scores = np.concatenate(scores)
		selected_idx = np.argsort(scores)[-self.stage_budget:]
		selected_image_ids = [self.unlabeled_set[idx] for idx in selected_idx]
		
		for image_id in selected_image_ids:
			domain = self.id2domain_mapping[image_id]
			self.stage_domain_labeled_set[domain].append(image_id)
		self.unlabeled_set = np.setdiff1d(np.array(self.unlabeled_set, dtype=np.int32), selected_image_ids).tolist()
		
		self._stage_finish()
		
		labeled_set = self.get_labeled_list_from_domain_labeled_set(self.domain_labeled_set)
		return labeled_set

