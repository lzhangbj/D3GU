import os
import wandb
import json

import numpy as np
from collections import defaultdict

from actmtda.utils.registry import SAMPLER
from actmtda.utils import all_logging_disabled


@SAMPLER.register("base_sampler")
class BaseSampler:
	def __init__(self, cfg,
					source_dataset,
					target_dataset,
					save_root,
					logger=None):
		self.cfg = cfg
		self.source_dataset = source_dataset
		self.target_dataset = target_dataset
		self.target_image_ids = [img[0] for img in self.target_dataset.imgs]
		self.save_root = save_root
		self.logger = logger
		
		# init from cfg
		self.num_class = cfg.DATASET.NUM_CLASS
		self.num_stage = cfg.SAMPLER.NUM_STAGE
		self.use_visualize = cfg.SAMPLER.VISUALIZE
		self.visualize_num_per_domain = cfg.SAMPLER.VISUALIZE_NUM_PER_DOMAIN
		self.n_target = len(cfg.DATASET.TARGET.DOMAIN_NAMES)
		self.domain_names = cfg.DATASET.TARGET.DOMAIN_NAMES
		self.data_root = cfg.DATASET.DATAROOT
		
		# read in
		self.class_names = []
		with open(cfg.DATASET.CLASS_LIST_PATH, 'r') as f:
			for line in f.readlines():
				self.class_names.append(line.strip())
		with open(cfg.DATASET.DATASET_MAPPING_PATH, 'r') as f:
			self.file2id_mapping = json.load(f)
		self.id2file_mapping = {int(val):key for key, val in self.file2id_mapping.items()}
		with open(cfg.DATASET.ID2CLASS_MAPPING_PATH, 'r') as f:
			id2class_mapping = json.load(f)
			self.id2class_mapping = {int(key):val for key, val in id2class_mapping.items()}
		with open(cfg.DATASET.ID2DOMAIN_MAPPING_PATH, 'r') as f:
			id2domain_mapping = json.load(f)
			self.id2domain_mapping = {int(key): val for key, val in id2domain_mapping.items()}
			
		self.domain_labeled_set = {domain:[] for domain in self.domain_names} # dict of domain to image id list
		self.stage_domain_labeled_set = {domain:[] for domain in self.domain_names}

		# init
		self.stage = 0
		self.unlabeled_set = [tup[0] for tup in target_dataset.imgs]
		
		if cfg.SAMPLER.STAGE_BUDGET > 1:
			self.stage_budget = int(cfg.SAMPLER.STAGE_BUDGET)
		else:
			self.stage_budget = int(len(self.unlabeled_set) * cfg.SAMPLER.STAGE_BUDGET)
		
		self.random_seed = self.cfg.RANDOM_SEED
	
	def get_image_idx(self, image_ids):
		# indices = np.where(np.in1d(np.array(self.target_image_ids, dtype=np.int32), np.array(image_ids, dtype=np.int32)))[0]
		x = np.array(self.target_image_ids).astype(np.int32)
		y = np.array(image_ids).astype(np.int32)
		
		xsorted = np.argsort(x)
		ypos = np.searchsorted(x[xsorted], y)
		indices = xsorted[ypos]
		return indices.tolist()
		
	def __call__(self, *args, **kwargs):
		return None
	
	def get_labeled_list_from_domain_labeled_set(self, domain_labeled_set):
		labeled_list = []
		for domain in self.domain_names:
			labeled_list += domain_labeled_set[domain]
		return labeled_list
	
	def _stage_init(self):
		# initialize
		np.random.seed(self.random_seed)
		self.stage += 1
		n_budget = self.stage_budget
		self.stage_domain_labeled_set = {domain: [] for domain in self.domain_names}
		
		self.logger.info(f"########## sample {n_budget} samples for stage-{self.stage} ###########")
	
	def _stage_finish(self):
		for domain, stage_domain_labeled_list in self.stage_domain_labeled_set.items():
			self.domain_labeled_set[domain] += stage_domain_labeled_list
		self.save_labeled_set()
		self.sample_summary()
		if self.use_visualize:
			self.visualize(self.visualize_num_per_domain)
	
	def save_labeled_set(self):
		stage_domain_labeled_set_save_file = os.path.join(self.save_root, f"stage-{self.stage}_stage_domain_labeled_set.json")
		with open(stage_domain_labeled_set_save_file, 'w') as f:
			json.dump(self.stage_domain_labeled_set, f)
			
		domain_labeled_set_save_file = os.path.join(self.save_root, f"stage-{self.stage}_domain_labeled_set.json")
		with open(domain_labeled_set_save_file, 'w') as f:
			json.dump(self.domain_labeled_set, f)
	
	def sample_summary(self):
		summary = defaultdict(int)
		
		max_domain_sample_num = 0
		min_domain_sample_num = 10000000
		max_class_sample_num = 0
		min_class_sample_num = 10000000
		for domain_idx, domain in enumerate(self.domain_names):
			domain_labeled_image_ids = np.array(self.domain_labeled_set[domain], dtype=np.int32)
			domain_labeled_image_classes = np.array([self.id2class_mapping[image_id] for image_id in domain_labeled_image_ids], dtype=np.int32)
			domain_sample_num = len(domain_labeled_image_ids)
			summary[f'sample_summary/domain-{domain}'] += domain_sample_num
			# record max and min domain smaple num
			if domain_sample_num > max_domain_sample_num:
				max_domain_sample_num = domain_sample_num
			if domain_sample_num < min_domain_sample_num:
				min_domain_sample_num = domain_sample_num
				
			for class_id in range(self.num_class):
				class_sample_num = (domain_labeled_image_classes==class_id).astype(np.int32).sum()
				summary[f'sample_summary/class-{self.class_names[class_id]}'] += class_sample_num
				if class_sample_num > max_class_sample_num:
					max_class_sample_num = class_sample_num
				if class_sample_num < min_class_sample_num:
					min_class_sample_num = class_sample_num
		
		max_stage_domain_sample_num = 0
		min_stage_domain_sample_num = 10000000
		max_stage_class_sample_num = 0
		min_stage_class_sample_num = 10000000
		for domain_idx, domain in enumerate(self.domain_names):
			stage_domain_labeled_image_ids = np.array(self.stage_domain_labeled_set[domain], dtype=np.int32)
			stage_domain_labeled_image_classes = np.array([self.id2class_mapping[image_id] for image_id in stage_domain_labeled_image_ids], dtype=np.int32)
			stage_domain_sample_num = len(stage_domain_labeled_image_ids)
			summary[f'stage_sample_summary/domain-{domain}'] += stage_domain_sample_num
			# record max and min stage domain smaple num
			if stage_domain_sample_num > max_stage_domain_sample_num:
				max_stage_domain_sample_num = stage_domain_sample_num
			if stage_domain_sample_num < min_stage_domain_sample_num:
				min_stage_domain_sample_num = stage_domain_sample_num
			
			for class_id in range(self.num_class):
				stage_class_sample_num = (stage_domain_labeled_image_classes==class_id).astype(np.int32).sum()
				summary[f'stage_sample_summary/class-{self.class_names[class_id]}'] += stage_class_sample_num
				if stage_class_sample_num > max_stage_class_sample_num:
					max_stage_class_sample_num = stage_class_sample_num
				if stage_class_sample_num < min_stage_class_sample_num:
					min_stage_class_sample_num = stage_class_sample_num
		
		summary = dict(summary)
		summary[f'sample_summary/domain-imb-ratio'] = float(min_domain_sample_num) / max_domain_sample_num
		summary[f'sample_summary/class-imb-ratio'] = float(min_class_sample_num) / max_class_sample_num
		summary[f'stage_sample_summary/domain-imb-ratio'] = float(min_stage_domain_sample_num) / max_stage_domain_sample_num
		summary[f'stage_sample_summary/class-imb-ratio'] = float(min_stage_class_sample_num) / max_stage_class_sample_num
		summary['stage'] = self.stage
		
		wandb.log(summary)
		
	def visualize(self, n_images_per_domain=3):
		with all_logging_disabled():
			for i, domain in enumerate(self.domain_names):
				domain_vis_num = min(len(self.stage_domain_labeled_set[domain]), n_images_per_domain)
				if domain_vis_num > 0:
					vis_image_ids = np.random.choice(self.stage_domain_labeled_set[domain], size=domain_vis_num, replace=False)
					vis_image_names = [self.id2file_mapping[int(image_id)] for image_id in vis_image_ids]
					vis_wandb_images = [
						wandb.Image(data_or_path=os.path.join(self.data_root, image_name),
						            caption=self.id2class_mapping[self.file2id_mapping[image_name]])
						for image_name in vis_image_names
					]
					wandb.log({f"vis/{domain}": vis_wandb_images, 'stage': self.stage})
	
		
	
