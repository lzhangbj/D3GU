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


@SAMPLER.register("target-combined_danngrad_sampler")
class TargetCombinedDANNGradSampler(BaseSampler):
	def __init__(self, *args, **kwargs):
		super(TargetCombinedDANNGradSampler, self).__init__(*args, **kwargs)
		self.apply_stage = self.cfg.SAMPLER.DANN.APPLY_STAGE
		self.is_domainwise = self.cfg.SAMPLER.DANN.IS_DOMAINWISE
		self.is_classwise = self.cfg.SAMPLER.DANN.IS_CLASSWISE
		self.agg = self.cfg.SAMPLER.DANN.AGG
		assert self.agg in [
			'cls-grad-scale',
			'cls-grad-scaleXoneplus-cos',
			'cls-grad-scaleXcos',
			
			'cos',  'dot', 'oneplus-cos',
			'entmin-oneplus-cos', 'entmin-cls-grad-scale', 'entmin-cls-grad-scaleXoneplus-cos',
			'cls-grad-scaleXabs-cos', 'cls-grad-scaleXminmax-cos',
			'cls-grad-scaleXoneplus-etgtprob-cos',
			'cls-grad-scaleXoneplus-clippedetgtprob-cos',
			'cls-grad-scaleXoneplus-abs-cos',
			'cls-grad-scaleXoneplus-neg-cos',
			'cls-grad-scaleXoneplus-sum-all-dom-cos',
			'cls-grad-scaleXoneplus-sum-tgt-dom-cos',
			'cls-grad-scaleXoneplus-avg-all-dom-cos',
			'cls-grad-scaleXoneplus-avg-tgt-dom-cos',
			'bvsb-cls-grad-scaleXoneplus-cos',
			'cls-grad-scaleXoneplus-entmax-cos',
			'cls-grad-scaleXoneplus-src-entmax-cos',
			'cls-grad-scaleXoneplus-dual-cos',
		                    ], self.agg
		self.normalization = self.cfg.SAMPLER.DANN.NORMALIZATION
		self.normalization_level = self.cfg.SAMPLER.DANN.NORMALIZATION_LEVEL
		assert self.normalization in ['max', 'minmax'], self.normalization
		assert self.normalization_level in ['all', 'domain', 'class', 'domain-class'], self.normalization_level
		self.T = self.cfg.SAMPLER.CLUE.SOFTMAX_T
		self.danngrad_weight = self.cfg.SAMPLER.DANN.DANNGRAD_WEIGHT
		self.entropy_weight = self.cfg.SAMPLER.DANN.ENTROPY_WEIGHT
		self.entmax_to_src_loss_weight = self.cfg.SAMPLER.DANN.ENTMAX_TO_SRC_LOSS_WEIGHT
		# self.class_dominant_gap = self.cfg.SAMPLER.CLASS_DOMINANT_GAP
		self.random_cand_max_num = self.cfg.SAMPLER.RANDOM_CAND_MAX_NUM
	
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
			cls_grads = cls_grads / np.linalg.norm(cls_grads, axis=1, keepdims=True)
			dann_grads = dann_grads / np.linalg.norm(dann_grads, axis=1, keepdims=True)
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
			cos_sim = (cos_sim + 1) / 2.
			dot_sim = cls_grad_norms * cos_sim
			danngrad_importance = dot_sim + 1e-8
		elif self.agg in ['cls-grad-scaleXoneplus-cos',
		                  'entmin-cls-grad-scaleXoneplus-cos',
		                  'cls-grad-scaleXoneplus-entmax-cos',
		                  'cls-grad-scaleXoneplus-src-entmax-cos',
		                  ]:
			cls_grad_norms = np.linalg.norm(cls_grads, axis=1)
			cls_grads = cls_grads / np.linalg.norm(cls_grads, axis=1, keepdims=True)
			dann_grads = dann_grads / np.linalg.norm(dann_grads, axis=1, keepdims=True)
			cos_sim = (cls_grads * dann_grads).sum(axis=1)
			dot_sim = cls_grad_norms * (1+cos_sim)
			danngrad_importance = dot_sim + 1e-8
		elif self.agg == 'cls-grad-scaleXoneplus-neg-cos':
			cls_grad_norms = np.linalg.norm(cls_grads, axis=1)
			cls_grads = cls_grads / np.linalg.norm(cls_grads, axis=1, keepdims=True)
			dann_grads = dann_grads / np.linalg.norm(dann_grads, axis=1, keepdims=True)
			cos_sim = (cls_grads * dann_grads).sum(axis=1)
			dot_sim = cls_grad_norms * (1-cos_sim)
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
		elif self.agg in ['cls-grad-scale', 'entmin-cls-grad-scale']:
			danngrad_importance = np.linalg.norm(cls_grads, axis=1) + 1e-8
		# multi domain gradients
		elif self.agg == 'cls-grad-scaleXoneplus-sum-all-dom-cos':
			cls_grad_norms = np.linalg.norm(cls_grads, axis=1)
			cls_grads = cls_grads / np.linalg.norm(cls_grads, axis=1, keepdims=True)
			dann_grads = dann_grads / np.linalg.norm(dann_grads, axis=2, keepdims=True)
			cos_sim = (cls_grads[:, None, :] * dann_grads).sum(axis=2)
			dot_sim = cls_grad_norms * (1 + cos_sim.sum(axis=1))
			danngrad_importance = dot_sim + 1e-8
		elif self.agg == 'cls-grad-scaleXoneplus-avg-all-dom-cos':
			cls_grad_norms = np.linalg.norm(cls_grads, axis=1)
			cls_grads = cls_grads / np.linalg.norm(cls_grads, axis=1, keepdims=True)
			dann_grads = dann_grads / np.linalg.norm(dann_grads, axis=2, keepdims=True)
			cos_sim = (cls_grads[:, None, :] * dann_grads).sum(axis=2)
			dot_sim = cls_grad_norms * (1 + cos_sim.mean(axis=1))
			danngrad_importance = dot_sim + 1e-8
		elif self.agg == 'cls-grad-scaleXoneplus-sum-tgt-dom-cos':
			cls_grad_norms = np.linalg.norm(cls_grads, axis=1)
			cls_grads = cls_grads / np.linalg.norm(cls_grads, axis=1, keepdims=True)
			dann_grads = dann_grads / np.linalg.norm(dann_grads, axis=2, keepdims=True)
			cos_sim = (cls_grads[:, None, :] * dann_grads[:, 1:]).sum(axis=2)
			dot_sim = cls_grad_norms * (1 + cos_sim.sum(axis=1))
			danngrad_importance = dot_sim + 1e-8
		elif self.agg == 'cls-grad-scaleXoneplus-avg-tgt-dom-cos':
			cls_grad_norms = np.linalg.norm(cls_grads, axis=1)
			cls_grads = cls_grads / np.linalg.norm(cls_grads, axis=1, keepdims=True)
			dann_grads = dann_grads / np.linalg.norm(dann_grads, axis=2, keepdims=True)
			cos_sim = (cls_grads[:, None, :] * dann_grads[:, 1:]).sum(axis=2)
			dot_sim = cls_grad_norms * (1 + cos_sim.mean(axis=1))
			danngrad_importance = dot_sim + 1e-8
		# bvsb
		elif self.agg == 'bvsb-cls-grad-scaleXoneplus-cos':
			cls_grad_norms = np.linalg.norm(cls_grads, axis=2)
			cls_grads = cls_grads / cls_grad_norms[:, :, None]
			dann_grads = dann_grads / np.linalg.norm(dann_grads, axis=1, keepdims=True)
			cos_sims = (cls_grads * dann_grads[:, None, :]).sum(axis=2)
			dot_sims = cls_grad_norms * (1+cos_sims)
			danngrad_importance = dot_sims[:, 0] - dot_sims[:, 1] + 1e-8
		# dual cos
		elif self.agg == 'cls-grad-scaleXoneplus-dual-cos':
			cls_grad_norms = np.linalg.norm(cls_grads, axis=1)
			normed_dann_grads = dann_grads / np.linalg.norm(dann_grads, axis=2, keepdims=True)
			normed_cls_grads = cls_grads / cls_grad_norms[:, None]
			dual_cosines = (normed_cls_grads[:, None, :] * normed_dann_grads).sum(axis=2)
			dual_cosines[:, 1] *= self.entmax_to_src_loss_weight
			dual_cosine_sums = dual_cosines.sum(axis=1)
			dot_sim = cls_grad_norms * (1 + dual_cosine_sums)
			danngrad_importance = dot_sim + 1e-8
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
		# assert np.all(np.logical_and(sample_weights>=0, sample_weights<=1)), [sample_weights.min(), sample_weights.max()]
		
		return sample_weights
	
	def __call__(self, model, discriminator=None, *args, **kwargs):
		self._stage_init()
		
		unlabeled_set = self.unlabeled_set
		# if len(unlabeled_set) > self.random_cand_max_num:
		# 	rand_cand_idx = np.random.permutation(len(unlabeled_set))[:self.random_cand_max_num]
		# 	unlabeled_set = [unlabeled_set[i] for i in rand_cand_idx]
			
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
			if 'bvsb-cls-grad-scale' in self.agg:
				top2_labels = torch.topk(logits, 2, dim=1, largest=True, sorted=True)[1]
				cls_grads_list = []
				for top_label_idx in range(2):
					loss = F.cross_entropy(logits, top2_labels[:, top_label_idx], reduction="sum")
					cls_grads = torch.autograd.grad(loss, feat, retain_graph=True)[0].detach().cpu().numpy()
					cls_grads_list.append(cls_grads)
				cls_grads = np.stack(cls_grads_list, axis=1)
			else:
				if 'entmin' in self.agg:
					loss = torch.special.entr(torch.softmax(logits, dim=1)).sum()
				else:
					loss = F.cross_entropy(logits, label, reduction="sum")
				cls_grads = torch.autograd.grad(loss, feat)[0].detach().cpu().numpy()
			# compute cls prob
			vanilla_scores = torch.softmax(logits.detach().cpu(), dim=1) + 1e-8
			cls_prob_list.append(vanilla_scores.detach().cpu().numpy())
			# compute entropy
			scores = torch.softmax(logits.detach().cpu() / self.T, dim=1) + 1e-8
			entropy = -(scores * torch.log(scores)).sum(1).cpu().numpy()
			# compute dom grad
			feat = feat.detach()
			feat.requires_grad_(True)
			dom_logits = discriminator(feat)
			dom_prob = torch.softmax(dom_logits.detach().cpu(), dim=1).numpy()
			if 'dom-cos' in self.agg:
				domains_dom_grads = []
				for domain_idx in range(self.n_target+1):
					aim_domains = torch.ones_like(domains) * domain_idx
					loss = F.cross_entropy(dom_logits, aim_domains, reduction='sum')
					dom_grads = torch.autograd.grad(loss, feat, retain_graph=True)[0].detach().cpu().numpy()
					domains_dom_grads.append(dom_grads)
				dom_grads = np.stack(domains_dom_grads, axis=1)
			elif 'src-entmax' in self.agg:
				src_domains = torch.zeros_like(domains)
				src_loss = F.cross_entropy(dom_logits, src_domains, reduction='sum')
				tgt_domain_probs = torch.softmax(dom_logits[:, 1:], dim=1)
				entmax_loss = - self.entmax_to_src_loss_weight * torch.special.entr(tgt_domain_probs).sum()
				loss = src_loss + entmax_loss
				dom_grads = torch.autograd.grad(loss, feat)[0].detach().cpu().numpy()
			elif 'dual-cos' in self.agg:
				src_domains = torch.zeros_like(domains)
				src_loss = F.cross_entropy(dom_logits, src_domains, reduction='sum')
				src_dom_grads = torch.autograd.grad(src_loss, feat, retain_graph=True)[0].detach().cpu().numpy()
				tgt_domain_probs = torch.softmax(dom_logits[:, 1:], dim=1)
				entmax_loss = - torch.special.entr(tgt_domain_probs).sum()
				entmax_grads = torch.autograd.grad(entmax_loss, feat, retain_graph=True)[0].detach().cpu().numpy()
				dom_grads = np.stack([src_dom_grads, entmax_grads], axis=1)
			elif 'entmax' in self.agg:
				domain_probs = torch.softmax(dom_logits, dim=1)
				loss = -torch.special.entr(domain_probs).sum()
				dom_grads = torch.autograd.grad(loss, feat)[0].detach().cpu().numpy()
			else:
				src_domains = torch.zeros_like(domains)
				loss = F.cross_entropy(dom_logits, src_domains, reduction='sum')
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
		
		selected_idx = np.argsort(sample_weights)[-self.stage_budget:]
		selected_image_ids = [unlabeled_set[idx] for idx in selected_idx]
		
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
